import io
import wave

import numpy as np
import pytest

from sargoshi.speaker.embed import pcm_duration_s, trim_silence
from sargoshi.speaker.enrol import Enroller, EnrolmentError
from sargoshi.speaker.identify import SpeakerIdentifier
from sargoshi.speaker.store import VoiceprintStore


async def _fresh(embedder, tmp_path):
    store = VoiceprintStore(str(tmp_path / "speakers.db"), embedding_model=embedder.model_id)
    await store.open(create_schema=True)
    ident = SpeakerIdentifier(embedder=embedder, store=store, threshold=0.70, top_k=5)
    enr = Enroller(embedder=embedder, store=store, identifier=ident, min_duration_s=0.4)
    return store, ident, enr


async def _enrolled(embedder, audio, tmp_path):
    """A store with two distinct-band speakers enrolled (Alice ~300, Bob ~1500)."""
    store, ident, enr = await _fresh(embedder, tmp_path)
    alice = await enr.enrol(
        name="Alice",
        audio=[audio.sine_wav(300, seed=1), audio.sine_wav(310, seed=2), audio.sine_wav(305, seed=3)],
        attributes={"gender": "female", "role": "resident"},
    )
    bob = await enr.enrol(
        name="Bob",
        audio=[audio.sine_wav(1500, seed=4), audio.sine_wav(1520, seed=5)],
        attributes={"gender": "male"},
    )
    return store, ident, enr, alice, bob


async def test_enroll_creates_profiles(fake_embedder, audio, tmp_path):
    _store, _ident, _enr, alice, bob = await _enrolled(fake_embedder, audio, tmp_path)
    assert alice.created and alice.embedding_count == 3
    assert bob.created and bob.embedding_count == 2


async def test_original_audio_stored_per_embedding(fake_embedder, audio, tmp_path):
    store, _ident, _enr, alice, _bob = await _enrolled(fake_embedder, audio, tmp_path)
    embs = await store.list_embeddings(alice.speaker_id)
    assert len(embs) == 3 and all(e.audio_bytes > 0 for e in embs)
    wav = await store.get_embedding_audio(embs[0].id)
    assert wav is not None and wav[:4] == b"RIFF"


async def test_identify_known_and_unknown(fake_embedder, audio, tmp_path):
    _store, ident, _enr, _alice, _bob = await _enrolled(fake_embedder, audio, tmp_path)

    r = await ident.identify(audio.sine_pcm(302, seed=99))
    assert r.known and r.name == "Alice" and r.confidence >= 0.70
    assert r.attributes.get("gender") == "female"
    assert len(r.candidates) == 2 and r.candidates[0].name == "Alice"

    r = await ident.identify(audio.sine_pcm(1510, seed=98))
    assert r.known and r.name == "Bob"

    r = await ident.identify(audio.sine_pcm(6000, seed=97))
    assert not r.known and r.speaker_id is None


async def test_add_to_existing_by_id_and_by_name(fake_embedder, audio, tmp_path):
    _store, _ident, enr, alice, _bob = await _enrolled(fake_embedder, audio, tmp_path)

    by_id = await enr.enrol(name="Alice", speaker_id=alice.speaker_id, audio=[audio.sine_wav(308, seed=6)])
    assert not by_id.created and by_id.embedding_count == 4

    by_name = await enr.enrol(name="Alice", audio=[audio.sine_wav(303, seed=10)])
    assert not by_name.created and by_name.speaker_id == alice.speaker_id
    assert by_name.embedding_count == 5


async def test_delete_embedding_recomputes_centroid(fake_embedder, audio, tmp_path):
    store, ident, _enr, alice, _bob = await _enrolled(fake_embedder, audio, tmp_path)
    embs = await store.list_embeddings(alice.speaker_id)
    affected = await store.delete_embedding(embs[0].id)
    assert affected == alice.speaker_id
    await ident.refresh()
    prof = await store.get_profile(alice.speaker_id)
    assert prof.embedding_count == 2 and prof.centroid is not None
    r = await ident.identify(audio.sine_pcm(302, seed=42))
    assert r.known and r.name == "Alice"


async def test_attributes_merge_on_reenroll(fake_embedder, audio, tmp_path):
    store, _ident, enr, alice, _bob = await _enrolled(fake_embedder, audio, tmp_path)
    await enr.enrol(
        name="Alice",
        speaker_id=alice.speaker_id,
        audio=[audio.sine_wav(306, seed=7)],
        attributes={"language": "en"},
    )
    prof = await store.get_profile(alice.speaker_id)
    assert prof.attributes == {"gender": "female", "role": "resident", "language": "en"}


async def test_similarity_matrix_diagonal_and_separation(fake_embedder, audio, tmp_path):
    _store, ident, _enr, _alice, _bob = await _enrolled(fake_embedder, audio, tmp_path)
    names, mat = await ident.similarity_matrix()
    assert set(names) == {"Alice", "Bob"} and mat.shape == (2, 2)
    assert np.allclose(np.diag(mat), 1.0, atol=1e-4)
    assert mat[0, 1] < 0.3  # different bands -> low cross-similarity


async def test_delete_profile_cascades(fake_embedder, audio, tmp_path):
    store, ident, _enr, _alice, bob = await _enrolled(fake_embedder, audio, tmp_path)
    assert await store.delete_profile(bob.speaker_id)
    assert await store.get_embeddings(bob.speaker_id) == []  # FK cascade removed rows
    await ident.refresh()
    r = await ident.identify(audio.sine_pcm(1510, seed=96))
    assert not r.known


async def test_empty_audio_is_rejected(fake_embedder, tmp_path):
    _store, _ident, enr = await _fresh(fake_embedder, tmp_path)
    with pytest.raises(EnrolmentError):
        await enr.enrol(name="X", audio=[])


async def test_any_wav_is_converted_and_the_converted_audio_stored(fake_embedder, audio, tmp_path):
    store, _ident, enr = await _fresh(fake_embedder, tmp_path)
    carol = await enr.enrol(name="Carol", audio=[audio.stereo_wav(400)])  # 44.1k stereo
    assert carol.created and carol.embedding_count == 1
    stored = await store.get_embedding_audio((await store.list_embeddings(carol.speaker_id))[0].id)
    with wave.open(io.BytesIO(stored), "rb") as w:
        assert (w.getframerate(), w.getsampwidth(), w.getnchannels()) == (16_000, 2, 1)


def test_trim_silence_trims_the_ends(audio):
    padded = audio.sine_pcm(0, dur=0.3) + audio.sine_pcm(400, dur=0.6, seed=8) + audio.sine_pcm(0, dur=0.3)
    trimmed = trim_silence(padded)
    assert pcm_duration_s(trimmed) < pcm_duration_s(padded)


async def test_model_tagging_prevents_cross_model_match(fake_embedder, audio, tmp_path):
    db = str(tmp_path / "spk.db")

    store_a = VoiceprintStore(db, embedding_model="modelA")
    await store_a.open(create_schema=True)
    id_a = SpeakerIdentifier(embedder=fake_embedder, store=store_a, threshold=0.70)
    enr_a = Enroller(embedder=fake_embedder, store=store_a, identifier=id_a, min_duration_s=0.4)
    dave = await enr_a.enrol(name="Dave", audio=[audio.sine_wav(700, seed=20), audio.sine_wav(710, seed=21)])
    await id_a.refresh()
    assert (await id_a.identify(audio.sine_pcm(705, seed=22))).name == "Dave"
    await store_a.close()

    # reopen the same DB under a different model -> Dave has no modelB voiceprints
    store_b = VoiceprintStore(db, embedding_model="modelB")
    await store_b.open(create_schema=True)
    id_b = SpeakerIdentifier(embedder=fake_embedder, store=store_b, threshold=0.70)
    assert await store_b.rebuild_centroids() == 0  # nothing usable under modelB
    await id_b.refresh()
    assert not (await id_b.identify(audio.sine_pcm(705, seed=23))).known
    embs = await store_b.list_embeddings(dave.speaker_id)
    assert len(embs) == 2 and all(e.model == "modelA" for e in embs)  # retained
    await store_b.close()


async def test_per_profile_model_tags(tmp_path):
    db = str(tmp_path / "tags.db")
    vec = np.ones(8, dtype=np.float32)

    s1 = VoiceprintStore(db, embedding_model="ecapa-tdnn")
    await s1.open(create_schema=True)
    prof = await s1.create_profile(name="Pieter")
    await s1.add_embeddings(prof.id, [(vec, b"RIFFaaaa")])
    assert (await s1.list_profiles())[0].models == ("ecapa-tdnn",)
    assert (await s1.get_profile(prof.id)).models == ("ecapa-tdnn",)
    await s1.close()

    s2 = VoiceprintStore(db, embedding_model="titanet-large")
    await s2.open(create_schema=True)
    await s2.add_embeddings(prof.id, [(vec * 0.5, b"RIFFbbbb")])
    assert (await s2.list_profiles())[0].models == ("ecapa-tdnn", "titanet-large")
    assert (await s2.get_by_name("Pieter")).models == ("ecapa-tdnn", "titanet-large")
    await s2.close()
