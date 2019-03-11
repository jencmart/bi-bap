import distutils.cygwinccompiler
import sys

def get_msvcr():
    """Include the appropriate MSVC runtime library if Python was built
    with MSVC 7.0 or later.
    """
    msc_pos = sys.version.find('MSC v.')
    if msc_pos != -1:
        msc_ver = sys.version[msc_pos+6:msc_pos+10]
    if msc_ver == '1300':
        # MSVC 7.0
        return ['msvcr70']
    elif msc_ver == '1310':
        # MSVC 7.1
        return ['msvcr71']
    elif msc_ver == '1400':
        # VS2005 / MSVC 8.0
        return ['msvcr80']
    elif msc_ver == '1500':
        # VS2008 / MSVC 9.0
        return ['msvcr90']
    elif msc_ver == '1600':
        # VS2010 / MSVC 10.0
        return ['msvcr100']
    elif msc_ver == '1700':
        # Visual Studio 2012 / Visual C++ 11.0
        return ['msvcr110']
    elif msc_ver == '1800':
        # Visual Studio 2013 / Visual C++ 12.0
        return ['msvcr120']
    elif msc_ver == '1900':
        # Visual Studio 2015 / Visual C++ 14.0
        # "msvcr140.dll no longer exists" http://blogs.msdn.com/b/vcblog/archive/2014/06/03/visual-studio-14-ctp.aspx
        return ['vcruntime140']
    else:
        # to do: can we make this futureproof?
        raise ValueError("Unknown MS Compiler version %s " % msc_ver)
distutils.cygwinccompiler.get_msvcr = get_msvcr
