"""Read .osu members from Sayo's public mini ZIP using bounded HTTP ranges.

The download route is used by Sayo's own frontend. Only directory records and
compressed chart byte ranges are requested; music/backgrounds are not extracted.
ZIP CRC and sizes are checked by zipfile; official MD5 is checked by the caller
when official per-difficulty metadata is available.
"""
import io
import re
import zipfile


class RemoteZip(io.RawIOBase):
    def __init__(self,sid,fetch,url=None):
        self.fetch=fetch
        self.url=url or f'https://txy1.sayobot.cn/beatmaps/download/mini/{sid}'
        raw,headers,url=fetch(self.url,headers={'Range':'bytes=-65557'},include_headers=True)
        match=re.fullmatch(r'bytes (\d+)-(\d+)/(\d+)',headers.get('Content-Range',''))
        if not match:raise ValueError('Sayo server did not honor ZIP byte range')
        a,b,self.size=map(int,match.groups())
        if len(raw)!=b-a+1:raise ValueError('Truncated ZIP directory range')
        self.url=url;self.etag=headers.get('ETag');self.position=0
        self.chunks=[(a,raw)];self.bytes_fetched=len(raw)

    def seekable(self):return True
    def readable(self):return True
    def tell(self):return self.position

    def seek(self,offset,whence=0):
        position=offset if whence==0 else self.position+offset if whence==1 else self.size+offset
        if position<0:raise ValueError('Negative ZIP seek')
        self.position=position;return position

    def block(self,a,b):
        for start,raw in self.chunks:
            if start<=a and b<=start+len(raw):return raw[a-start:b-start]
        if b-a>8*1024*1024:raise ValueError('Chart ZIP range exceeds size limit')
        headers={'Range':f'bytes={a}-{b-1}'}
        if self.etag:headers['If-Range']=self.etag
        raw,response,_=self.fetch(self.url,headers=headers,include_headers=True)
        if response.get('Content-Range')!=f'bytes {a}-{b-1}/{self.size}' or len(raw)!=b-a:
            raise ValueError('ZIP changed or server ignored requested byte range')
        if self.etag and response.get('ETag')!=self.etag:raise ValueError('ZIP revision changed')
        self.chunks.append((a,raw));self.bytes_fetched+=len(raw)
        return raw

    def read(self,size=-1):
        end=self.size if size<0 else min(self.size,self.position+size)
        if end<=self.position:return b''
        raw=self.block(self.position,end);self.position=end;return raw

    def prefetch(self,infos):
        spans=[]
        for info in sorted(infos,key=lambda f:f.header_offset):
            a=info.header_offset
            b=min(self.size,a+30+len(info.filename.encode('utf-8'))+len(info.extra)+info.compress_size+1024)
            if spans and a-spans[-1][1]<=8192 and b-spans[-1][0]<=8*1024*1024:
                spans[-1][1]=max(spans[-1][1],b)
            else:spans.append([a,b])
        for a,b in spans:self.block(a,b)


def fetch_charts(sid,fetch):
    remote=RemoteZip(sid,fetch)
    with zipfile.ZipFile(remote) as archive:
        infos=[f for f in archive.infolist() if f.filename.lower().endswith('.osu') and not f.is_dir()]
        if len(infos)>2000 or any(f.file_size>8*1024*1024 for f in infos):
            raise ValueError('Unexpected chart ZIP member count or size')
        remote.prefetch(infos)
        charts=[(info.filename,archive.read(info)) for info in infos]
    return charts,dict(url=remote.url,archive_etag=remote.etag,archive_bytes=remote.size,
                       bytes_fetched=remote.bytes_fetched,chart_members=len(charts))
