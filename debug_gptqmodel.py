
try:
    import gptqmodel.utils.nogil_patcher
    print("Module loaded successfully")
    print(dir(gptqmodel.utils.nogil_patcher))
    
    if hasattr(gptqmodel.utils.nogil_patcher, '_get_config_for_key'):
        print("Found _get_config_for_key in module")
    else:
        print("_get_config_for_key NOT found in module")
        
    # Check if Autotuner is available
    if hasattr(gptqmodel.utils.nogil_patcher, 'Autotuner'):
         print("Found Autotuner class")
    
except ImportError as e:
    print(f"ImportError: {e}")
except Exception as e:
    print(f"Error: {e}")
