{
  In this unit, important structures of GPU are defined.
  TDllFunction is the signature for methods inside a DLL.
  Only TDllFunctions can be managed by the PluginManager.
  
  TStack is the internal structure used by plugins to communicate
  with the GPU core.
   
  (c) by 2002-2010 the GPU Development Team
  (c) by 2010 HB9TVM, Freedom Light my Fire
  This unit is released under GNU Public License (GPL)
}
unit stacks;

interface

uses gpuconstants, formatsets,
     SysUtils;

const
    GPU_FLOAT_STKTYPE   = 10;
    GPU_BOOLEAN_STKTYPE = 20;
    GPU_STRING_STKTYPE  = 30;
  
type TGPUFloat = Extended;  // type for floats on stack
type TGPUStackTypes = Array [1..MAX_STACK_PARAMS] of Longint;

type TGPUError = record
    ErrorID : Longint;
    ErrorMsg,            // the error in human readable form
    ErrorArg : String;  // some parameter for the error
end;

type
  TStack = record
    stack    : Array [1..MAX_STACK_PARAMS] of TGPUFloat;
    strStack : Array [1..MAX_STACK_PARAMS] of String;
    stkType  : TGPUStackTypes;
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


type TDescFunction = function : String;
type PDescFunction = ^TDescFunction;

// initialization and conversion functions
procedure initStack(var stk : TStack);
function  stackToStr(var stk : TStack; var error : TGPUError) : String;
function  gpuTypeToStr(gputype : Longint) : String;

// check functions
function maxStackReached(var stk : TStack; var error : TGPUError) : Boolean; 
function isEmptyStack(var stk : TStack) : Boolean;
function enoughParametersOnStack(required : Longint; var stk : TStack; var error : TGPUError) : Boolean;
function typeOfParametersCorrect(required : Longint; var stk : TStack; var types : TGPUStackTypes; var error : TGPUError) : Boolean;

// loading stuff on stack
function pushStr  (str : String; var stk : TStack; var error : TGPUError) : Boolean;
function pushFloat(float : TGPUFloat; var Stk : TStack; var error : TGPUError) : Boolean;
function pushBool (b : boolean; var Stk : TStack; var error : TGPUError) : Boolean;

// checking stack types
function isGPUFloat  (i : Longint; var stk : TStack) : Boolean;
function isGPUBoolean(i : Longint; var stk : TStack) : Boolean;
function isGPUString (i : Longint; var stk : TStack) : Boolean;

// popping stuff from stack
function popFloat(var float : TGPUFloat; var stk : TStack; var error : TGPUError) : Boolean;
function popBool (var b : Boolean; var stk : TStack; var error : TGPUError) : Boolean;
function popStr  (var str : String; var stk : TStack; var error : TGPUError) : Boolean;


implementation

procedure initStack(var stk : TStack);
var i : Longint;
begin
 stk.Idx := 0; // to have it empty
 for i:=1 to MAX_STACK_PARAMS do
   begin
     stk.Stack[i] := 0;
     stk.StrStack[i] := '';
     stk.stkType[i] := GPU_FLOAT_STKTYPE;
   end;

end;

function stackToStr(var stk : TStack; var error : TGPUError) : String;
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
           str := str + ', ' + FloatToStr(stk.Stack[i]);
	   end
	 else
	   if stk.stkType[i]=GPU_BOOLEAN_STKTYPE then
           begin
             if stk.Stack[i]>0 then
			   str := str + ', true'
			 else
                           str := str + ', false';
           end
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

function gpuTypeToStr(gputype : Longint) : String;
begin
 if (gputype = GPU_FLOAT_STKTYPE) then
    Result := 'float'
 else
 if (gputype = GPU_BOOLEAN_STKTYPE) then
    Result := 'boolean'
 else
 if (gputype = GPU_FLOAT_STKTYPE) then
    Result := 'string'
 else
   raise Exception.Create('Unknown type in gpuTypeToStr (stacks.pas)');
end;

function typeOfParametersCorrect(required : Longint; var stk : TStack; var types : TGPUStackTypes; var error : TGPUError) : Boolean;
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
			  error.errorArg := 'Required type for parameter '+IntToStr(i)+' was '+gpuTypeToStr(types[i])
			                    + ' but type on stack was '+gpuTypeToStr(stk.stkType[stk.Idx-required+i]);
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

function pushFloat(float : TGPUFloat; var stk : TStack; var error : TGPUError) : Boolean;
var hasErrors : Boolean;
begin
 Result := false;
 Inc(stk.Idx);
 hasErrors := maxStackReached(stk, error); 
 if not hasErrors then
                 begin                     
                   Stk.Stack[stk.Idx] := float;
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


function isEmptyStack(var stk : TStack) : Boolean;
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
 Result := (stk.stkType[i]=GPU_BOOLEAN_STKTYPE);
end;

function isGPUString(i : Longint; var stk : TStack) : Boolean;
begin
 if (i<1) or (i>MAX_STACK_PARAMS) then raise Exception.Create('Index out of range in isGPUString ('+IntToStr(i)+')');
 Result := (stk.stkType[i]=GPU_STRING_STKTYPE);
end;

function popFloat(var float : TGPUFloat; var stk : TStack; var error : TGPUError) : Boolean;
var types : TGPUStackTypes;
begin
  Result  := false;
  types[1]:= GPU_FLOAT_STKTYPE;
  if not typeOfParametersCorrect(1, stk,  types, error) then Exit;
  float := stk.Stack[stk.Idx];
  Dec(stk.Idx);
  Result := true;
end;

function popBool(var b : Boolean; var stk : TStack; var error : TGPUError) : Boolean;
var types : TGPUStackTypes;
begin
  Result  := false;
  types[1]:= GPU_BOOLEAN_STKTYPE;
  if not typeOfParametersCorrect(1, stk,  types, error) then Exit;
  b := (stk.Stack[stk.Idx]>0);
  Dec(stk.Idx);
  Result := true;
end;

function popStr(var str : String; var stk : TStack; var error : TGPUError) : Boolean;
var types : TGPUStackTypes;
begin
  Result  := false;
  types[1]:= GPU_STRING_STKTYPE;
  if not typeOfParametersCorrect(1, stk,  types, error) then Exit;
  str := stk.strStack[stk.Idx];
  Dec(stk.Idx);
  Result := true;
end;

end.
