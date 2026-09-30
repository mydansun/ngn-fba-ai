"""Create project-private mTLS material; refuses to replace existing keys."""
import argparse
import os
import re
from pathlib import Path
import secrets
import shutil
import subprocess

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--output', type=Path, required=True)
p.add_argument('--server', required=True)
a = p.parse_args()
if not re.fullmatch(r"[A-Za-z0-9._:-]+", a.server):
    p.error("Invalid relay host")
os.umask(0o077)
a.output.mkdir(parents=True, exist_ok=False)
root = a.output.resolve()
(root/'pki').mkdir()

def run(*args):
    subprocess.run(['openssl', *args], cwd=root/'pki', stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, check=True)

run('req','-x509','-newkey','rsa:3072','-nodes','-days','3650','-subj','/CN=ngn-fba-ai-ca',
    '-addext','basicConstraints=critical,CA:TRUE','-addext','keyUsage=critical,keyCertSign,cRLSign',
    '-keyout','ca.key','-out','ca.crt')
token = secrets.token_urlsafe(48)
for role, cn, usage in [('frps','fba-ai.internal','serverAuth'),('frpc','fba-ai-gpu','clientAuth')]:
    dest=root/role;dest.mkdir()
    run('req','-new','-newkey','rsa:3072','-nodes','-subj',f'/CN={cn}','-keyout',str(dest/'tls.key'),'-out',f'{role}.csr')
    (root/'pki'/f'{role}.ext').write_text(f'basicConstraints=critical,CA:FALSE\nkeyUsage=critical,digitalSignature,keyEncipherment\nextendedKeyUsage={usage}\nsubjectAltName=DNS:{cn}\n')
    run('x509','-req','-in',f'{role}.csr','-CA','ca.crt','-CAkey','ca.key','-CAcreateserial','-days','365',
        '-extfile',f'{role}.ext','-out',str(dest/'tls.crt'))
    shutil.copyfile(root/'pki/ca.crt',dest/'ca.crt')
    (dest/'token').write_text(token)
common='''auth.method = "token"
auth.tokenSource.type = "file"
auth.tokenSource.file.path = "/etc/frp/token"
auth.additionalScopes = ["HeartBeats", "NewWorkConns"]
transport.tls.certFile = "/etc/frp/tls.crt"
transport.tls.keyFile = "/etc/frp/tls.key"
transport.tls.trustedCaFile = "/etc/frp/ca.crt"
log.to = "console"
log.level = "info"
'''
(root/'frps/frps.toml').write_text('''bindAddr = "0.0.0.0"
bindPort = 7000
transport.tls.force = true
allowPorts = [{start=50052, end=50054}]
maxPortsPerClient = 3
'''+common)
client=f'''serverAddr = "{a.server}"
serverPort = 1201
loginFailExit = false
transport.tls.enable = true
transport.tls.serverName = "fba-ai.internal"
transport.tls.disableCustomTLSFirstByte = true
'''+common
for port, service in enumerate(('ocr','layout','clean'),50052):
    client+=f'''\n[[proxies]]
name = "ngn-fba-{service}"
type = "tcp"
localIP = "{service}"
localPort = 50051
remotePort = {port}
'''
(root/'frpc/frpc.toml').write_text(client)
(root/'api_key').write_text(secrets.token_urlsafe(48))
print('Created private certificates and configuration; no secret values printed')
