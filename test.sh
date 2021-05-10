make -j 8 && ./ffmpeg -y -i ~/bbb.webm -filter_complex "[0:a][0:v]addetect=server_address=api.imernest.com:server_port=8000:application_id=spotz:context_id=testz" -f null -
