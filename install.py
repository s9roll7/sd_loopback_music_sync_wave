import launch


if not launch.is_installed("librosa"):
	launch.run_pip("install librosa", "requirements for sd_loopback_music_sync_wave")

if not launch.is_installed("pydub"):
	launch.run_pip("install pydub", "requirements for sd_loopback_music_sync_wave")

