{
  In this unit, important structures of GPU are defined.
  TDllFunction is the signature for methods inside a DLL.
  Only TDllFunctions can be managed by the PluginManager.
  
  TStack is the internal structure used by plugins to communicate
  with the GPU core.
   
}
unit stacks;

interface

const
    GPU_FLOAT_STKTYPE   = 10;
	GPU_BOOLEAN_STKTYPE = 20;
	GPU_STRING_STKTYPE  = 30;
  
type TGPUFloat = Extended;  // type for floats on stack
type TGPUType  = Longint;   // type for id identifing types on stack
type TGPUStackType = Array [1..MAX_STACK_PARAMS] of TGPUType;

type TGPUError = record
    ErrorID : Longint;
    ErrorMsg,            // the error in human readable form
    ErrorArg : String;  // some parameter for the error
end;

type
  TStack = record
    stack    : Array [1..MAX_STACK_PARAMS] of TGPUFloat;
    strStack : Array [1..MAX_STACK_PARAMS] of String;
    stkType  : TGPUStackType;
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
  PDllFunction = ^TDllFunction;
type

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

// initialization and conversion functions
function initStack(var stk : TStack);
function stackToStr(var stk : Stack; var error : TGPUError) : String;

// check functions
function maxStackReached(var stk : TStack; var error : TGPUError) : Boolean; 
function isEmptyStack(var stk : stk : TStack) : Boolean;
function enoughParametersOnStack(required : Longint; var stk : TStack; var error : TGPUError) : Boolean;
function typeOfParametersCorrect(required : Longint; var stk : TStack; var types : TGPUStackType; var error : TGPUError) : Boolean;

// loading stuff on stack
function pushStr  (str : String; var stk : TStack; var error : TGPUError) : Boolean;
function pushFloat(float : TGPUFloat; var Stk : TStack; var error : TGPUError) : Boolean;
function pushBool (b : boolean; var Stk : TStack; var error : TGPUError) : Boolean;

// checking stack types
function isGPUFloat  (i : Longint; var stk : TStack) : Boolean;
function isGPUBoolean(i : Longint; var stk : TStack) : Boolean;
function isGPUString (i : Longint, var stk : TStack) : Boolean;

// popping stuff from stack
function popFloat(var float : TGPUFloat; var stk : Stack; var error : TGPUError) : Boolean;
function popBool (var b : Boolean; var stk : Stack; var error : TGPUError) : Boolean;
function popStr  (var str : String; var stk : Stack; var error : TGPUError) : Boolean;


implementation

function initStack(var stk : TStack);
var i : Longint;
begin
 stk.Idx := 0; // to have it empty
 for i:=1 to MAX_STACK_PARAM do
   begin
     stk.Stack[i] := 0;
     stk.StrStack[i] := '';
     stk.stkType[i] := GPU_FLOAT_STKTYPE;
   end;

end;

function stackToStr(var stk : Stack; var error : TGPUError) : String;
var i : Longint;
    str : String;
begin
 str := '';
 for i:=1 to stk.Idx do
      begin
        if stk.stkType[i]=GPU_STRING_STKTYPE then
           begin
             // we need to add a string
             str := str + ', '+ QUOTE + stk.StrStack[i] + QUOTE;             
           end
         else
		 if stk.stkType[i]=GPU_FLOAT_STKTYPE then
         begin
		   // we need to add a float
           str := str + ', ' + FloatToStr(stk.Stack[i], formatSet.fs);
		 end
		 else
		 if stk.stkType[i]=GPU_BOOLEAN_STKTYPE then
           begin
             if stk.Stack[i]>0 then
			   str := str + ', true';
			 else
               str := str + ', false';			 
           end;
         else
           begin
		     // in this way we mantain backward compatibility
             error.errorID  := UNKNOWN_STACK_TYPE_ID ;
			 error.errorMsg := UNKNOWN_STACK_TYPE;
             error.errorArg := IntToStr(stk.stkType[i]);			 
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

function enoughParametersOnStack(required : Longint; var stk : TStack; var error : TGPUError) : Boolean;
begin
 Result := false;
 if (required<1) or (required>MAX_STACK_PARAMS) then raise Exception.Create('Required parameter out of range in enoughParametersOnStack ('+IntToStr(required)+')');
 if stk.Idx<required then
       begin
	    error.errorID  := NOT_ENOUGH_PARAMETERS_ID;
		error.errorMsg := NOT_ENOUGH_PARAMETERS;
		error.errorArg := 'Required: '+IntToStr(required)+' Available: '+IntToStr(stk.Idx);
	   end
	 else Result := true;  
end;

function typeOfParametersCorrect(required : Longint; var stk : TStack; var types : TGPUStackType; var error : TGPUError) : Boolean;
var i : Longint;
begin
  Result := false;
  if not enoughParametersOnStack(required, stk, error) then Exit;
  for i:=1 to required do
      begin
	    if types[i]<>stk.stkType[stk.Idx-required+i] then
		    begin
			  error.errorID  := WRONG_TYPE_PARAMETERS_ID;
			  error.errorMsg := WRONG_TYPE_PARAMETERS;
			  error.errorArg := 'Required type: '+IntToStr(types[i])+' but stack type was: '+IntToStr(stk.stkType[stk.Idx-required+i]);
			  Exit;
			end;
	  end;
  
  Result := true;
end;



function pushStr(str : String; var Stk : TStack; var error : TGPUError) : Boolean;
var hasErrors : Boolean;
begin
 Result := false;
 Inc(stk.Idx);
 hasErrors := maxStackReached(stk, error); 
 if not hasErrors then
                 begin                     
                   Stk.Stack[stk.Idx] := 0;
                   Stk.StrStack[stk.Idx] := str;
                   Stk.stkType[stk.Idx] := GPU_STRING_STKTYPE;
                   Result := true;
                 end;              
end;

function pushFloat(ext : TGPUFloat; var Stk : TStack; var error : TGPUError) : Boolean;
var hasErrors : Boolean;
begin
 Result := false;
 Inc(stk.Idx);
 hasErrors := maxStackReached(stk, error); 
 if not hasErrors then
                 begin                     
                   Stk.Stack[stk.Idx] := ext;
                   Stk.StrStack[stk.Idx] := '';
                   Stk.stkType[stk.Idx] := GPU_FLOAT_STKTYPE;
                   Result := true;
                 end;              
end;

function pushBool(b : boolean; var Stk : TStack; var error : TGPUError) : Boolean;
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
                   Stk.stkType[stk.Idx] := GPU_BOOLEAN_STKTYPE;
                   Result := true;
                 end;              
end;


function isEmptyStack(var stk : stk : TStack) : Boolean;
begin
 Result := (stk.Idx = 0);
end;

function isGPUFloat(i : Longint; var stk : TStack) : Boolean;
begin
 if (i<1) or (i>MAX_STACK_PARAMS) then raise Exception.Create('Index out of range in isGPUFloat ('+IntToStr(i)+')');
 Result := (stk.stkType[i]=GPU_FLOAT_STKTYPE);
end;

function isGPUBoolean(i : Longint; var stk : TStack) : Boolean;
begin
 if (i<1) or (i>MAX_STACK_PARAMS) then raise Exception.Create('Index out of range in isGPUBoolean ('+IntToStr(i)+')');
 Result := (stk.stkType[i]=GPU_FLOAT_BOOLEAN);
end;

function isGPUString(i : Longint, var stk : TStack) : Boolean;
begin
 if (i<1) or (i>MAX_STACK_PARAMS) then raise Exception.Create('Index out of range in isGPUString ('+IntToStr(i)+')');
 Result := (stk.stkType[i]=GPU_STRING_STKTYPE);
end;

function popFloat(var float : TGPUFloat; var stk : Stack; var error : TGPUError) : Boolean;
var types : TGPUTypes;
begin
  Result  := false;
  types[1]:= GPU_FLOAT_STKTYPE;
  if not typeOfParametersCorrect(1, stk,  types, error) then Exit;
  float := stk.Stack[stk.Idx];
  Dec(stk.Idx);
  Result := true;
end;

function popBool(var b : Boolean; var stk : Stack; var error : TGPUError) : Boolean;
var types : TGPUTypes;
begin
  Result  := false;
  types[1]:= GPU_BOOLEAN_STKTYPE;
  if not typeOfParametersCorrect(1, stk,  types, error) then Exit;
  b := (stk.Stack[stk.Idx]>0);
  Dec(stk.Idx);
  Result := true;
end;

function popString(var str : String; var stk : Stack; var error : TGPUError) : Boolean;
var types : TGPUTypes;
begin
  Result  := false;
  types[1]:= GPU_STRING_STKTYPE;
  if not typeOfParametersCorrect(1, stk,  types, error) then Exit;
  str := stk.strStack[stk.Idx];
  Dec(stk.Idx);
  Result := true;
end;

end.
