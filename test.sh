make -j 8 && ./ffmpeg -y -i ~/cctv1.mp4 -vf "addetect=server_address=localhost:server_port=5000:application_id=test_app:context_id=test_context" -f null -
