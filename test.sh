make -j 8
export GRPC_DEFAULT_SSL_ROOTS_FILE_PATH=isrgrootx1.pem
./ffmpeg_g -y -loglevel debug -i ~/vids/sintel-1280-surround.mp4 -filter_complex_script filter.txt -map 0 -b:v 2M -hls_playlist_type vod ~/hls/924375f050b6420fec8e36d546b63aba_stereo/index.m3u8

run -y -i ~/vids/movies.mkv -filter_complex_script filter.txt -map 0 -b:v 2M -hls_playlist_type vod ~/hls/throwaway/index.m3u8

./ffmpeg -y -i ~/vids/day_of_reckoning.stereo.mp4 -b:v 2M -hls_playlist_type vod ~/hls/924375f050b6420fec8e36d546b63aba_stereo/index.m3u8 