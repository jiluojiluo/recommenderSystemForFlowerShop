'''
  undone,TODO
'''


import socket

HOST, PORT = '127.0.0.1', 19999
listen_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
listen_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
listen_socket.bind((HOST, PORT))
listen_socket.listen(1)
print('Serving HTTP on port %s ...' % PORT)
while True:
  client_connection, client_address = listen_socket.accept()
  request = client_connection.recv(1024)

  print(bytes.decode(request))

  http_response = bytes('HTTP/1.1 200 OK Hello, World!',encoding='utf8')


  client_connection.sendall(http_response)
  client_connection.close()

