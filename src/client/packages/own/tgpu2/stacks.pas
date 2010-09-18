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
  MAXSTACK           = 128;          // Maximum size of Stack in Virtual Machine
  MAX_COLLECTING_IDS = 128;          // Maximum number of Jobs we keep also average track
  WRONG_PACKET_ID    = 7777777;      // if you get seven seven as result, the plugin returns false
  INF                = 1.0 / 0.0;    // infinite to distinguish Strings from floats
  QUOTE              = Chr(39);      // alias for apostrophe, '
  WRONG_GPU_COMMAND = 'UNKNOWN_COMMAND';


type
  TStack = record
    stack : Array [1..MAXSTACK] of Extended;
    Idx      : Longint;     //  Index on Stack where Operations take place
                            //  if Idx is 0 the stack is empty
    Progress : Extended;    //  indicates plugin progress from 0 to 100}
    
    {  Stack for strings, only for
              Borland DLLs. If a value in stack is INF, then StrStack
              is assigned to a String. If a value in StrStack is NULL,
              then a float is assigned in Stack.

              There is only  one stack pointer for both Stack and StrStack.}
    StrStack: array[1..MAXSTACK] of PChar;
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
    LastResult: Extended;
    N:        : Longint;
    Sum,                    {with sum and N we can compute average}
    Min,
    Max,
    Avg    : Extended;  
  end;

implementation


end.
