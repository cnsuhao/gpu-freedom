{
  In this unit, important structures of GPU are defined.
  TDllFunction is the signature for methods inside a DLL.
  Only TDllFunctions can be managed by the PluginManager.
  
  TStack is the internal structure used by plugins to communicate
  with the GPU core.
   
  (c) by 2002-2010 the GPU Development Team
  (c) by 2010 HB9TVM, "freedom light my fire!"
  This unit is released under GNU Public License (GPL)
}
unit stacks;

interface

uses gpuconstants, formatsets,
     SysUtils;

const
    NO_STKTYPE      = 0;     // dummy if not typed
    FLOAT_STKTYPE   = 10;
    BOOLEAN_STKTYPE = 20;
    STRING_STKTYPE  = 30;
  
type TStkFloat  = Extended;  // type for floats on stack
type TStkString = String;    // type for strings on stack
type TStkBoolean = Boolean;  // type for booleans on stack
type TStkArgType = Longint; // type distinguishing types on stack

type TStkTypes = Array [1..MAX_STACK_PARAMS] of TStkArgType;

type TStkError = record
    ErrorID : Longint;
    ErrorMsg,            // the error in human readable form
    ErrorArg : String;  // some parameter for the error
end;

type
  TStack = record
    stack    : Array [1..MAX_STACK_PARAMS] of TStkFloat;
    strStack : Array [1..MAX_STACK_PARAMS] of TStkString;
    stkType  : TStkTypes;
    Idx      : Longint;     //  Index on Stack where Operations take place
                            //  if Idx is 0 the stack is empty
    
    Progress : TStkFloat;    //  indicates plugin progress from 0 to 100}
   
    
    workunitIncoming,            // filename of workunit placed in directory workunits/staged
    workunitOutgoing : String;   // if the plugin outputs something, it will create a file in
                                 // workunits/outgoing directory
    error : TStkError;           // contains errors if there are any
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
function  stackToStr(var stk : TStack) : String;
function  stkTypeToStr(stktype : TStkArgType) : String;

// check functions
function maxStackReached(var stk : TStack) : Boolean;
function isEmptyStack(var stk : TStack) : Boolean;
function enoughParametersOnStack(required : Longint; var stk : TStack) : Boolean;
function typeOfParametersCorrect(required : Longint; var stk : TStack; var types : TStkTypes) : Boolean;

// loading stuff on stack
function pushStr  (str : TStkString; var stk : TStack) : Boolean;
function pushFloat(float : TStkFloat; var Stk : TStack) : Boolean;
function pushBool (b : TStkBoolean; var Stk : TStack) : Boolean;

// checking stack types
function isStkFloat  (i : Longint; var stk : TStack) : Boolean;
function isStkBoolean(i : Longint; var stk : TStack) : Boolean;
function isStkString (i : Longint; var stk : TStack) : Boolean;

// popping stuff from stack
function popFloat(var float : TStkFloat; var stk : TStack) : Boolean;
function popBool (var b : TStkBoolean; var stk : TStack) : Boolean;
function popStr  (var str : TStkString; var stk : TStack) : Boolean;


implementation

procedure initStack(var stk : TStack);
var i : Longint;
begin
 stk.Idx := 0; // to have it empty
 for i:=1 to MAX_STACK_PARAMS do
   begin
     stk.Stack[i] := 0;
     stk.StrStack[i] := '';
     stk.stkType[i] := NO_STKTYPE;
   end;

end;

function stackToStr(var stk : TStack) : String;
var i   : Longint;
    str : String;
begin
 str := '';
 for i:=1 to stk.Idx do
      begin
        if stk.stkType[i]=STRING_STKTYPE then
           begin
             // we need to add a string
             str := str + ', '+ QUOTE + stk.StrStack[i] + QUOTE;             
           end
         else
	   if stk.stkType[i]=FLOAT_STKTYPE then
           begin
	   // we need to add a float
           str := str + ', ' + FloatToStr(stk.Stack[i]);
	   end
	 else
	   if stk.stkType[i]=BOOLEAN_STKTYPE then
           begin
             if stk.Stack[i]>0 then
			   str := str + ', true'
			 else
                           str := str + ', false';
           end
         else
           begin
		     // in this way we mantain backward compatibility
             stk.error.errorID  := UNKNOWN_STACK_TYPE_ID ;
	     stk.error.errorMsg := UNKNOWN_STACK_TYPE;
             stk.error.errorArg := IntToStr(stk.stkType[i]);
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


function maxStackReached(var stk : TStack) : Boolean;
begin
 Result := false;
 if (stk.Idx>MAX_STACK_PARAMS) then
       begin
         Result := true;
         stk.error.errorID := TOO_MANY_ARGUMENTS_ID;
         stk.error.errorMsg := TOO_MANY_ARGUMENTS;
         stk.error.errorArg := '';
         Dec(stk.Idx);
       end;
end;

function enoughParametersOnStack(required : Longint; var stk : TStack) : Boolean;
begin
 Result := false;
 if (required<0) or (required>MAX_STACK_PARAMS) then raise Exception.Create('Required parameter out of range in enoughParametersOnStack ('+IntToStr(required)+')');
 if stk.Idx<required then
       begin
	    stk.error.errorID  := NOT_ENOUGH_PARAMETERS_ID;
	    stk.error.errorMsg := NOT_ENOUGH_PARAMETERS;
	    stk.error.errorArg := 'Required: '+IntToStr(required)+' Available: '+IntToStr(stk.Idx);
       end
	 else Result := true;  
end;

function stkTypeToStr(stktype : TStkArgType) : String;
begin
 if (stktype = FLOAT_STKTYPE) then
    Result := 'float'
 else
 if (stktype = BOOLEAN_STKTYPE) then
    Result := 'boolean'
 else
 if (stktype = FLOAT_STKTYPE) then
    Result := 'string'
 else
   raise Exception.Create('Unknown type in stkTypeToStr (stacks.pas)');
end;

function typeOfParametersCorrect(required : Longint; var stk : TStack; var types : TStkTypes) : Boolean;
var i : Longint;
begin
  Result := false;
  if not enoughParametersOnStack(required, stk) then Exit;
  for i:=1 to required do
      begin
	    if types[i]<>stk.stkType[stk.Idx-required+i] then
		    begin
			  stk.error.errorID  := WRONG_TYPE_PARAMETERS_ID;
			  stk.error.errorMsg := WRONG_TYPE_PARAMETERS;
			  stk.error.errorArg := 'Required type for parameter '+IntToStr(i)+' was '+stkTypeToStr(types[i])
			                      + ' but type on stack was '+stkTypeToStr(stk.stkType[stk.Idx-required+i]);
			  Exit;
			end;
	  end;
  
  Result := true;
end;



function pushStr(str : TStkString; var Stk : TStack) : Boolean;
var hasErrors : Boolean;
begin
 Result := false;
 Inc(stk.Idx);
 hasErrors := maxStackReached(stk);
 if not hasErrors then
                 begin                     
                   Stk.Stack[stk.Idx] := 0;
                   Stk.StrStack[stk.Idx] := str;
                   Stk.stkType[stk.Idx] := STRING_STKTYPE;
                   Result := true;
                 end;              
end;

function pushFloat(float : TStkFloat; var stk : TStack) : Boolean;
var hasErrors : Boolean;
begin
 Result := false;
 Inc(stk.Idx);
 hasErrors := maxStackReached(stk);
 if not hasErrors then
                 begin                     
                   Stk.Stack[stk.Idx] := float;
                   Stk.StrStack[stk.Idx] := '';
                   Stk.stkType[stk.Idx] := FLOAT_STKTYPE;
                   Result := true;
                 end;              
end;

function pushBool(b : TStkBoolean; var stk : TStack) : Boolean;
var hasErrors : Boolean;
    value     : TStkFloat;
begin
 Result := false;
 Inc(stk.Idx);
 if b then value :=1 else value := 0;
 hasErrors := maxStackReached(stk);
 if not hasErrors then
                 begin                     
                   Stk.Stack[stk.Idx] := value;
                   Stk.StrStack[stk.Idx] := '';
                   Stk.stkType[stk.Idx] := BOOLEAN_STKTYPE;
                   Result := true;
                 end;              
end;


function isEmptyStack(var stk : TStack) : Boolean;
begin
 Result := (stk.Idx = 0);
end;

function isStkFloat(i : Longint; var stk : TStack) : Boolean;
begin
 if (i<1) or (i>MAX_STACK_PARAMS) then raise Exception.Create('Index out of range in isGPUFloat ('+IntToStr(i)+')');
 Result := (stk.stkType[i]=FLOAT_STKTYPE);
end;

function isStkBoolean(i : Longint; var stk : TStack) : Boolean;
begin
 if (i<1) or (i>MAX_STACK_PARAMS) then raise Exception.Create('Index out of range in isGPUBoolean ('+IntToStr(i)+')');
 Result := (stk.stkType[i]=BOOLEAN_STKTYPE);
end;

function isStkString(i : Longint; var stk : TStack) : Boolean;
begin
 if (i<1) or (i>MAX_STACK_PARAMS) then raise Exception.Create('Index out of range in isGPUString ('+IntToStr(i)+')');
 Result := (stk.stkType[i]=STRING_STKTYPE);
end;

function popFloat(var float : TStkFloat; var stk : TStack) : Boolean;
var types : TStkTypes;
begin
  Result  := false;
  types[1]:= FLOAT_STKTYPE;
  if not typeOfParametersCorrect(1, stk,  types) then Exit;
  float := stk.Stack[stk.Idx];
  Dec(stk.Idx);
  Result := true;
end;

function popBool(var b : TStkBoolean; var stk : TStack) : Boolean;
var types : TStkTypes;
begin
  Result  := false;
  types[1]:= BOOLEAN_STKTYPE;
  if not typeOfParametersCorrect(1, stk,  types) then Exit;
  b := (stk.Stack[stk.Idx]>0);
  Dec(stk.Idx);
  Result := true;
end;

function popStr(var str : TStkString; var stk : TStack) : Boolean;
var types : TStkTypes;
begin
  Result  := false;
  types[1]:= STRING_STKTYPE;
  if not typeOfParametersCorrect(1, stk,  types) then Exit;
  str := stk.strStack[stk.Idx];
  Dec(stk.Idx);
  Result := true;
end;

end.
