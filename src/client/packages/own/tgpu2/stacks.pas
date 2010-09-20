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
    
    workunitIncoming,            // filename of workunit placed in directory workunits/staged
    workunitOutgoing : String;   // if the plugin outputs something, it will create a file in
                                 // workunits/outgoing directory
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
