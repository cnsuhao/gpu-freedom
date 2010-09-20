{
  In this unit, important structures of GPU are defined.
  TDllFunction is the signature for methods inside a DLL.
  Only TDllFunctions can be managed by the PluginManager.
  
  TStack is the internal structure used by plugins to communicate
  with the GPU core.
   
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
    stack    : Array [1..MAX_STACK_PARAMS] of TGPUFloat;
    strStack : Array [1..MAX_STACK_PARAMS] of String;
    isFloat  : Array [1..MAX_STACK_PARAMS] of boolean;
    Idx      : Longint;     //  Index on Stack where Operations take place
                            //  if Idx is 0 the stack is empty
    
    Progress : TGPUFloat;    //  indicates plugin progress from 0 to 100}
   
    
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

// TODO: move this in collectresults
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

function initStack(var stk : TStack);
function stackToStr(var stk : Stack) : String;

function maxStackReached(var stk : TStack; var error : TGPUError) : Boolean; 
function LoadStringOnStack(str : String; var stk : TStack; var error : TGPUError) : Boolean;
function LoadExtendedOnStack(ext : TGPUFloat; var Stk : TStack; var error : TGPUError) : Boolean;
function LoadBooleanOnStack(b : boolean; var Stk : TStack; var error : TGPUError) : Boolean;


implementation

function initStack(var stk : TStack);
var i : Longint;
begin
 stk.Idx := 0; // to have it empty
 for i:=1 to MAX_STACK_PARAM do
   begin
     stk.Stack[i] := 0;
     stk.StrStack[i] := '';
     stk.isFloat[i] := true;
   end;

end;

function stackToStr(var stk : Stack) : String;
var i : Longint;
    str : String;
begin
 str := '';
 for i:=1 to stk.Idx do
      begin
        if not stk.isFloat[i] then
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


function maxStackReached(var stk : TStack; var error : TGPUError) : Boolean; 
begin
 Result := false;
 if (stk.Idx>MAX_STACK_PARAMS) then
       begin
         Result := true;
         error.errorID := TOO_MANY_ARGUMENTS_ID;
         error.errorMsg := TOO_MANY_ARGUMENTS;
         error.errorArg := '';
         Dec(stk.Idx);
       end;
end;


function LoadStringOnStack(str : String; var Stk : TStack; var error : TGPUError) : Boolean;
var hasErrors : Boolean;
begin
 Result := false;
 Inc(stk.Idx);
 hasErrors := maxStackReached(stk, error); 
 if not hasErrors then
                 begin                     
                   Stk.Stack[stk.Idx] := 0;
                   Stk.StrStack[stk.Idx] := str;
                   Stk.isFloat[stk.Idx] := false;
                   Result := true;
                 end;              
end;

function LoadExtendedOnStack(ext : TGPUFloat; var Stk : TStack; var error : TGPUError) : Boolean;
var hasErrors : Boolean;
begin
 Result := false;
 Inc(stk.Idx);
 hasErrors := maxStackReached(stk, error); 
 if not hasErrors then
                 begin                     
                   Stk.Stack[stk.Idx] := ext;
                   Stk.StrStack[stk.Idx] := '';
                   Stk.isFloat[stk.Idx] := true;
                   Result := true;
                 end;              
end;

function LoadBooleanOnStack(b : boolean; var Stk : TStack; var error : TGPUError) : Boolean;
var hasErrors : Boolean;
    value     : TGPUFloat;
begin
 Result := false;
 Inc(stk.Idx);
 if b then value :=1 else value := 0;
 hasErrors := maxStackReached(stk, error);
 if not hasErrors then
                 begin                     
                   Stk.Stack[stk.Idx] := value;
                   Stk.StrStack[stk.Idx] := '';
                   Stk.isFloat[stk.Idx] := true;
                   Result := true;
                 end;              
end;


end.
