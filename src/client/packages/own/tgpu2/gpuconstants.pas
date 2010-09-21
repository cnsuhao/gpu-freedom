unit gpuconstants;
{
  In this class constants for the GPU core v2 are defined.
  Error codes and important parameters can be found here as well.
}
interface

const
  DEFAULT_THREADS = 3;   // default number of threads in TGPU2.pas
  MAX_THREADS     = 16;  // maximum number of allowed threads in TGPU2.pas

  MAX_STACK_PARAMS   = 128;          // Maximum size of Stack in Virtual Machine
                                     // changes in this value or in TStack structure
                                     // require recompilation of all plugins
                                     
  MAX_COLLECTING_IDS = 128;          // Maximum number of Jobs we keep also average track

  QUOTE              = Chr(39);      // alias for apostrophe, ', used in argretrievers.pas
  
  // error codes, used gpu core wide
  NO_ERROR_ID              = 0;
  NO_ERROR                 = 'NO ERROR';
  METHOD_NOT_FOUND_ID      = 1;
  METHOD_NOT_FOUND         = 'METHOD NOT FOUND';
  EMPTY_ARGUMENT_ID        = 2;
  EMPTY_ARGUMENT           = 'EMPTY ARGUMENT';
  MISSING_QUOTE_ID         = 3;
  MISSING_QUOTE            = 'ENDING QUOTE MISSING ('+QUOTE+')';
  COULD_NOT_PARSE_FLOAT_ID = 4;
  COULD_NOT_PARSE_FLOAT    = 'COULD NOT PARSE FLOAT';
  WRONG_NUMBER_OF_BRACKETS_ID = 5;
  WRONG_NUMBER_OF_BRACKETS = 'WRONG NUMBER OF BRACKETS';
  TOO_MANY_ARGUMENTS_ID    = 6;
  TOO_MANY_ARGUMENTS       = 'TOO MANY ARGUMENTS';
  PLUGIN_THREW_EXCEPTION_ID= 7;
  PLUGIN_THREW_EXCEPTION   = 'PLUGIN_THREW_EXCEPTION';
  NO_AVAILABLE_THREADS_ID  = 8;
  NO_AVAILABLE_THREADS     = 'NO AVAILABLE THREADS';
  NOT_ENOUGH_PARAMETERS_ID = 9;
  NOT_ENOUGH_PARAMETERS    = 'NOT ENOUGH PARAMETERS';
  WRONG_TYPE_PARAMETERS_ID = 10;
  WRONG_TYPE_PARAMETERS    = 'WRONG TYPE OF PARAMETERS';
  UNKNOWN_STACK_TYPE_ID    = 11;
  UNKNOWN_STACK_TYPE       = 'UNKNOWN STACK TYPE';
  
  // constants used in argretrievers.pas and parsers.pas
  GPU_ARG_UNKNOWN      = 0;
  GPU_ARG_CALL         = 10;
  GPU_ARG_STRING       = 20;
  GPU_ARG_FLOAT        = 30;
  GPU_ARG_EXPRESSION   = 40;
  GPU_ARG_BOOLEAN      = 50;
  
  GPU_ARG_SPECIAL_CALL_PLUGIN   = 60;
  GPU_ARG_SPECIAL_CALL_FRONTEND = 70;
  GPU_ARG_SPECIAL_CALL_NODE     = 80;
  GPU_ARG_SPECIAL_CALL_USER     = 90;
  GPU_ARG_SPECIAL_CALL_THREAD   = 100;
  GPU_ARG_SPECIAL_CALL_RESULT   = 110;
  GPU_ARG_SPECIAL_CALL_CORE     = 120;
  
  GPU_ARG_ERROR        = 999;
  
  

implementation


end.