
# TAGT VALIDATION ERROR REPORT
**Generated**: 2025-06-21 02:22:48
**Status**: VALIDATION FAILED

## Error Details
'charmap' codec can't encode character '\u274c' in position 401: character maps to <undefined>

## Stack Trace
Traceback (most recent call last):
  File "C:\Users\ADMIN\OneDrive\Desktop\SLE\validation_plan\run_full_validation.py", line 388, in main
    final_report = validator.save_results()
  File "C:\Users\ADMIN\OneDrive\Desktop\SLE\validation_plan\run_full_validation.py", line 354, in save_results
    f.write(report)
    ~~~~~~~^^^^^^^^
  File "C:\Users\ADMIN\AppData\Local\Programs\Python\Python313\Lib\encodings\cp1252.py", line 19, in encode
    return codecs.charmap_encode(input,self.errors,encoding_table)[0]
           ~~~~~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
UnicodeEncodeError: 'charmap' codec can't encode character '\u274c' in position 401: character maps to <undefined>


## Completed Phases
{'data_audit': 'FAILED', 'baseline_models': 'FAILED', 'tagt_validation': 'FAILED', 'comprehensive_analysis': 'COMPLETED'}

## Recommendation
Review error details and ensure all dependencies are properly installed and data files are accessible.
