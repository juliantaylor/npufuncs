def configuration(parent_package='', top_path=None):
    import numpy
    from numpy.distutils.misc_util import Configuration

    config = Configuration('.',
                           parent_package,
                           top_path)
    config.add_extension('npfma', ['fma.c', 'fma-avx.c', 'fma-fma3.c', 'fma-fma4.c', 'fma-sse2.c'], extra_compile_args=['-std=c99', '-O3', '-ffast-math', '-ffp-contract=fast'])

    return config

if __name__ == "__main__":
    from numpy.distutils.core import setup
    setup(configuration=configuration)
