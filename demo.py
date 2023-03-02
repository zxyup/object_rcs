# import base64
# str = base64.b64encode(b'1')
# print(str,type(str))
# print(str.decode('utf8'))
 
# str = base64.b64decode(str)

# print(str,type(str))
# a = str.decode('utf8')
# print(a,type(a))

from transmissions.serial import Serial
s = Serial('d')
s.start()
while 1:
    data=s.read()
    print(data)