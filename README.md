npufuncs
========

numpy ufuncs

numpy fused multiply and add testing

```
 python setup.py build_ext --inplace
 
 irunner --ipython test.ipy 
 ```

 Requires gcc >= 4.7.
 
 edit the `set_type` to try different variants, if the cpu does not support it it will crash.
 * fma 4 is AMD >= bulldozer

 * fma 3 is intel AVX2
