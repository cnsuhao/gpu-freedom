{
  In this unit, important structures of GPU are defined.
  TDllFunction is the signature for methods inside a DLL.
  Only TDllFunctions can be managed by the PluginManager.
  
  TStack is the internal structure used by plugins to communicate
  with the GPU core.
  
  TGPUCollectResult is used by the GPU component to collect
  results internally. However, frontends should collect
  results themselves. They cannot access this structure.
  
}
unit stacks;


interface

const
  MAX_STACK_PARAM    = 128;          // Maximum size of Stack in Virtual Machine
  MAX_COLLECTING_IDS = 128;          // Maximum number of Jobs we keep also average track
  INF                = 1.0 / 0.0;    // infinite to distinguish Strings from floats
  QUOTE              = Chr(39);      // alias for apostrophe, '
  
  // error codes
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
  
type TGPUFloat : Extended;

type TGPUError = record
    ErrorID : Longint;
    ErrorMsg,            // the error in human readable form
    ErrorArg : String;  // some parameter for the error
end;

type
  TStack = record
    stack    : Array [1..MAXSTACK] of TGPUFloat;
    Idx      : Longint;     //  Index on Stack where Operations take place
                            //  if Idx is 0 the stack is empty
    Progress : TGPUFloat;    //  indicates plugin progress from 0 to 100}
    
    {  Stack for strings, only for
              Freepascal/Borland DLLs. If a value in stack is INF, then StrStack
              is assigned to a String. 
          }
    StrStack: array[1..MAXSTACK] of String;
  end;

type  
  PStack = ^TStack;
  
type
  TDllFunction = function(var stk: TStack): boolean;
type
  PDllFunction = ^TDllFunction;

type TDescFunction = function : String;
     PDescFunction = ^TDescFunction;


type   {here we collect results, computing average and so on}
  TGPUCollectResult = record
    TotalTime: TDateTime;
    FirstResult,
    LastResult: TGPUFloat;
    N:        : Longint;
    Sum,                    {with sum and N we can compute average}
    Min,
    Max,
    Avg    : TGPUFloat;  
  end;

function stackToStr(var stk : Stack) : String;

implementation

function stackToStr(var stk : Stack) : String;
var i : Longint;
    str : String;
begin
 str := '';
 for i:=1 to stk.Idx do
      begin
        if stk.Stack[i] = INF then
           begin
             // we need to add a string
             str := str + ', '+ QUOTE + stk.StrStack[i] + QUOTE;             
           end
         else
           begin
             // we need to add a float
             str := str + ', ' + FloatToStr(stk.Stack[i], formatSet.fs);
           end;           
      end;
      
 if (str<>'') then
       begin
         // we remove the last two chars ', ' from the string
         if Pos(', ', str) = 1 then
            Delete(str, 1, 2);
       end;       
 Result := str;
end;

end.
