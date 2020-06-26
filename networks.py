#import socket
from socket import *

'''target_host = "www.google.com"
target_port = 80

#create socket
client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

#connect the client
client.connect((target_host, target_port))

#send some data
client.send(("GET / HTTP/1.1\r\nHost: google.com\r\n\r\n").encode('utf-8'))

#receive some data
response = client.recv(4096).decode('utf-8')

print(response)'''

s = socket(AF_INET, SOCK_STREAM)
s.connect(("www.python.org", 80))
s.send("GET /index.html HTTP/1.0\n\n".encode('utf-8'))
data = s.recv(10000).decode('utf-8')
print(data)
s.close()
