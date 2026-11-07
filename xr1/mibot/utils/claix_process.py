"""Linux process identity, including start ticks to reject PID reuse."""
import os
import fcntl
import stat
from pathlib import Path


def process_identity(pid=None):
    pid = os.getpid() if pid is None else int(pid)
    fields = Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()
    # state is field 3, starttime is field 22. Zombies cannot supervise a run.
    if fields[0] == 'Z':
        raise ProcessLookupError('Controller is a zombie')
    return f'{pid}:{fields[19]}'


def process_alive(identity):
    try:
        return process_identity(int(identity.split(':')[0])) == identity
    except (OSError, ValueError, IndexError):
        return False


def writer_lifetime_lock(root, *, probe=False):
    """Rank-owned lock survives controller/torchrun death, until writer drainage.

    Launcher probes it only while holding its existing root lock. Never unlink
    lock files: another process may still hold the inode.
    """
    path = Path(root)/'.checkpoint-writer.lock'
    if probe and not os.path.lexists(path):
        return None
    fd = os.open(path, os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW | os.O_CLOEXEC, 0o600)
    try:
        if not stat.S_ISREG(os.fstat(fd).st_mode):
            raise RuntimeError('Writer lock must be a regular file')
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        stream = os.fdopen(fd, 'a')
    except BaseException:
        os.close(fd)
        raise
    if probe:
        stream.close()
        return None
    return stream
