"""Enrichment of the song corpus from open, ban-free sources.

Submodules:
    match: pure text normalization + matching keys (doctested).
    musicbrainz: MusicBrainz API client — version-level first-release dating
        and the work/recording version graph (the remake spine).
    genius_kaggle: import the Hungarian subset of the Genius lyrics dump.

MusicBrainz is the authoritative source for *when a song was first published*
(per recording), and for grouping versions of the same composition by work;
zeneszoveg and Genius attach lyrics to those versions.
"""

# MusicBrainz asks every client to identify itself with a contactable UA.
USER_AGENT = "music_networks/0.1 (https://github.com/crow-intelligence/music_networks)"
