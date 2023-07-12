import logging
import time
import functools

def pyciemss_logging_wrappper( function):
    def wrapped(*args, **kwargs):
        try:
            start_time = time.perf_counter()
            result = function(*args, **kwargs)
            end_time = time.perf_counter()
            logging.info(
                f"Elapsed time for {function.__name__}:",
                end_time - start_time
                )
            return result
        except Exception as e:

            log_message = f"""
                ###############################

                There was an exception in pyciemss
                
                Function name: {function.__name__}

                Docs : {function.__doc__}

                ################################
            """
            logging.exception(log_message)
            raise e
    functools.update_wrapper(wrapped, function)
    return wrapped
