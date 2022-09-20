import os
import sys

import io
from pathlib import Path
import select
from shutil import rmtree
import subprocess as sp
import sys
from typing import Dict, Tuple, Optional, IO


# Customize the following options!
model = "mdx_extra"
extensions = ["mp3", "wav", "ogg", "flac", "m4a"]  # we will look for all those file types.


# Options for the output audio.
mp3 = False
mp3_rate = 320
float32 = True  # output as float 32 wavs, unsused if 'mp3' is True.
# You cannot set both `float32 = True` and `int24 = True` !!


from os import path

basepath = path.dirname(__file__)

#filepath = path.abspath(path.join(basepath, "..", "..", "fileIwantToOpen.txt"))
#f = open(filepath, "r")

stems = sys.argv[1:]

#track_path = basepath + "/separated/mdx_extra/" + stem + "/"
#audio_path = basepath + "/" + stem


in_path = basepath
    

out_path = in_path + "/separated/"


def find_files(in_path):
    out_2 = []
    for file in Path(in_path).iterdir():
        if file.suffix.lower().lstrip(".") in extensions:
            out_2.append(file)
    out = []
    for file in out_2:
        for stem in stems:
            if str(file).find(stem) != -1:
                print ("Found!")
                print(file)
                out.append(file)
    return out

#def copy_process_streams(process: sp.Popen):
#    def raw(stream: Optional[IO[bytes]]) -> IO[bytes]:
#        assert stream is not None
#        if isinstance(stream, io.BufferedIOBase):
#            stream = stream.raw
#        return stream
#
#    p_stdout, p_stderr = raw(process.stdout), raw(process.stderr)
#    stream_by_fd: Dict[int, Tuple[IO[bytes], io.StringIO, IO[str]]] = {
#        p_stdout.fileno(): (p_stdout, sys.stdout),
#        p_stderr.fileno(): (p_stderr, sys.stderr),
#    }
#    fds = list(stream_by_fd.keys())
#
#    while fds:
#        # `select` syscall will wait until one of the file descriptors has content.
##        ready, _, _ = select.select(fds, [], [])
##        for fd in ready:
#            p_stream, std = stream_by_fd[fd]
#            raw_buf = p_stream.read(2 ** 16)
#            if not raw_buf:
#                fds.remove(fd)
#                continue
#            buf = raw_buf.decode()
#            std.write(buf)
#            std.flush()
            
            
def separate(inp=None, outp=None):
    inp = inp or in_path
    outp = outp or out_path
    cmd = ["python", "-m", "demucs", "-o", str(outp), "-n", model]
    if mp3:
        cmd += ["--mp3", f"--mp3-bitrate={mp3_rate}"]
    if float32:
        cmd += ["--float32"]
#    if int24:
#        cmd += ["--int24"]
#    if two_stems is not None:
#        cmd += [f"--two-stems={two_stems}"]
    files = [str(f) for f in find_files(inp)]
    if not files:
        print(f"No valid audio files in {in_path}")
        return
    print("Going to separate the files:")
    print('\n'.join(files))
    print("With command: ", " ".join(cmd))
    
    from subprocess import check_output
    check_output(cmd + files, shell=True).decode()
    #p = sp.Popen(cmd + files, stdout=sp.PIPE, stderr=sp.PIPE)
    #copy_process_streams(p)
    #p.wait()
    #if p.returncode != 0:
    #    print("Command failed, something went wrong.")


#separate()

if __name__ == "__main__":
    import os
    import sys
    
    import io
    from pathlib import Path
    import select
    from shutil import rmtree
    import subprocess as sp
    import sys
    from typing import Dict, Tuple, Optional, IO