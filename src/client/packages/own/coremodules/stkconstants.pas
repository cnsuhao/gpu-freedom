unit stkconstants;
{
  In this class constants for the GPU core v2 are defined.
  Error codes and important parameters can be found here as well.
  
   (c) by 2002-2010 the GPU Development Team
  (c) by 2010 HB9TVM
  This unit is released under GNU Public License (GPL)
}
interface

const
  GPU_CORE_VERSION = '1.0.0';

  DEFAULT_THREADS = 3;   // default number of threads in TGPU2.pas
  MAX_THREADS     = 16;  // maximum number of allowed threads in TGPU2.pas

  MAX_STACK_PARAMS   = 256;          // Maximum size of Stack in Virtual Machine
                                     // changes in this value or in TStack structure
                                     // require recompilation of all plugins
                                     
  MAX_RESULTS = 128;          // Maximum number of jobs we keep track
  MAX_RESULTS_FOR_ID = 64;   // Maximum number of job results we remember for an id

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
  COULD_NOT_LOAD_PLUGIN_ID    = 12;
  COULD_NOT_LOAD_PLUGIN       = 'COULD NOT LOAD PLUGIN';
  COULD_NOT_DISCARD_PLUGIN_ID = 13;
  COULD_NOT_DISCARD_PLUGIN    = 'COULD NOT DISCARD PLUGIN';
  STILL_NO_RESULTS_ID         = 14;
  STILL_NO_RESULTS            = 'STILL NO RESULTS';
  MAX_NUMBER_OF_PLUGINS_REACHED_ID = 15;
  MAX_NUMBER_OF_PLUGINS_REACHED    = 'MAXIMUM NUMBER OF PLUGINS REACHED';
  INDEX_OUT_OF_RANGE_ID    = 16;
  INDEX_OUT_OF_RANGE       = 'INDEX OUT OF RANGE';
  COULD_NOT_PARSE_POINTER_ID = 17;
  COULD_NOT_PARSE_POINTER    = 'COULD NOT PARSE POINTER';
  
  // constants used in argretrievers.pas and parsers.pas
  STK_ARG_UNKNOWN      = 0;
  STK_ARG_CALL         = 10;
  STK_ARG_STRING       = 20;
  STK_ARG_FLOAT        = 30;
  STK_ARG_EXPRESSION   = 40;
  STK_ARG_BOOLEAN      = 50;
  STK_ARG_POINTER      = 60;
  
  STK_ARG_SPECIAL_CALL_PLUGIN   = 100;
  STK_ARG_SPECIAL_CALL_FRONTEND = 110;
  STK_ARG_SPECIAL_CALL_NODE     = 120;
  STK_ARG_SPECIAL_CALL_USER     = 130;
  STK_ARG_SPECIAL_CALL_THREAD   = 140;
  STK_ARG_SPECIAL_CALL_RESULT   = 150;
  STK_ARG_SPECIAL_CALL_CORE     = 160;
  
  STK_ARG_ERROR        = 999;
  
  INF = 1.0/0.0; // used to retrieve minimums
  
  

implementation


end.
