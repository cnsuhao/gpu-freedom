unit argretrievers;
{
This unit defines the object TArgRetriever which is initialized
with a GPU command. 

The object TArgRetrieve allows to iterate over the arguments of the command
with hasArguments() and getArgument().

It is used by the TGPUParser to parse the arguments

  (c) by 2002-2010 the GPU Development Team
  (c) by 2010 HB9TVM
  This unit is released under GNU Public License (GPL)
}
interface

uses SysUtils,
     stacks, plugins, utils, formatsets, gpuconstants, specialcommands;

type TArgStk = record
     argtype   : TStkArgType;
     argvalue  : TStkFloat;
     argstring : TStkString;
end;

type TArgRetriever = class(TObject)
 public 
   constructor Create(job : String); overload;
   constructor Create(job : String; var speccommands : TSpecialCommand); overload;
   destructor Destroy();
   function getJob() : String;
   
   function hasArguments() : Boolean;
   function getArgument(var error : TStkError) : TArgStk;
   
 private
   job_,
   toparse_      : String;
   speccommands_ : TSpecialCommand;
   
   procedure deleteComma; 
   function getBracketArgument(openbracket, closebracket : String; 
                               errorID : Longint; errorMsg : String; 
                               var error : TStkError) : TArgStk;
   function getStringArgument(var error : TStkError) : TArgStk;
   function getFloatArgument(var error : TStkError) : TArgStk;
   function getExpressionArgument(openbracket : String; var error : TStkError) : TArgStk;

   function getOtherArgument(var error : TStkError) : TArgStk;
   function getSpecialArgument(var error : TStkError; arg : TStkString; specialType : TStkArgType) : TArgStk;
   function getBooleanArgument(var error : TStkError;arg : TStkString) : TArgStk;
   function getCallArgument(var error : TStkError; arg : TStkString) : TArgStk;
   
end;


implementation

constructor TArgRetriever.Create(job : String); overload;
begin
 inherited Create();
 job_ := Trim(job);
 toparse_ := job_;
 speccommands_ := nil;
end;

constructor TArgRetriever.Create(job : String; var speccommands : TSpecialCommand); overload;
begin
 Create(job);
 speccommands_ := speccommands;
end;

destructor TArgRetriever.Destroy();
begin
 if Assigned(speccommands_) then speccommands_.Free;
end;

function TArgRetriever.getJob() : String;
begin
  Result := job_;
end;

function TArgRetriever.hasArguments() : Boolean;
begin
  Result := toparse_<> '';
end;

function TArgRetriever.getArgument(var error : TStkError) : TArgStk;
var startchar : Char;
    ordchar   : ShortInt;
begin
 if toparse_= '' then raise Exception.Create('getArgument() called when there are no arguments!');
 toparse_ := Trim(toparse_);

 startchar := toparse_[1];
 ordchar := Ord(startchar);
 if startchar = ',' then
    begin
       // an empty argument, raise error
       Result.argtype := STK_ARG_ERROR;
       error.ErrorID  := EMPTY_ARGUMENT_ID;
       error.ErrorMsg := EMPTY_ARGUMENT;
       error.ErrorArg := toparse_;
       Exit;
    end
 else   
 // if it starts with a quote is a string
 if startchar = QUOTE then
     Result := getStringArgument(error)
 else    
 // if it is a number
 if (ordchar>=Ord('0')) and (ordchar<=Ord('9')) then
     Result := getFloatArgument(error)
 else
 if (startchar = '{') or (startchar = '(')  then
     Result := getExpressionArgument(startchar, error)
 else    
 // it has to be something else
     Result := getOtherArgument(error);
end;

procedure TArgRetriever.deleteComma();
begin
 //delete an eventual comma
 toparse_ := TrimLeft(toparse_);
 if Pos(',', toparse_) = 1 then
      Delete(toparse_, 1, 1);
end;

function TArgRetriever.getBracketArgument(openbracket, closebracket : String; errorID : Longint; errorMsg : String; 
                            var error : TStkError) : TArgStk;
var 
   bracketCount, i : Longint;
   arg : String;
begin
    
    //string begins with bracket, we will return an argument
    //enclosed between two brackets
    bracketCount := 1;
    i := 2;
    arg := '';

    while (bracketCount <> 0) and (i <= Length(toparse_)) do
    begin
      if toparse_[i] = openbracket then
        Inc(BracketCount)
      else
      if toparse_[i] = closebracket then
        Dec(BracketCount);
      Inc(i);
    end;

    if bracketCount = 0 then
    begin
      arg := Copy(toparse_, 2, i - 3); // i is at length+1, additionally we remove the brackets
      // i-1 because we increment when we find
      // the right closing bracket
      Delete(toparse_, 1, i - 1);
      deleteComma();
      Result.argtype := STK_ARG_EXPRESSION;
      Result.argstring := arg;
    end
    else {problem in brackets}
    begin
      Result.argtype   := STK_ARG_ERROR;
      error.errorID := errorID;
      error.errorMsg := errorMsg;
      error.errorArg := toparse_;
    end;
end;

function TArgRetriever.getStringArgument(var error : TStkError) : TArgStk;
var i : Longint;
    arg : String;
begin
 i := 2;
 while (toparse_[i] <> QUOTE) and (i < Length(toparse_)) do
      Inc(i);

 if (i=Length(toparse_)) and (toparse_[i]<>QUOTE) then
     begin
       // quote is not closed
       Result.argtype := STK_ARG_ERROR;
       error.errorID := MISSING_QUOTE_ID;
       error.errorMsg := MISSING_QUOTE;
       error.errorArg := toparse_;
       Exit;
     end;     
 
 if (i=2) then
   arg := '' // empty string
 else   
   arg := Copy(toparse_, 2, i-2); // copy takes the number of chars not an index
 
 Delete(toparse_, 1, i);
 deleteComma();
 
 Result.argtype   := STK_ARG_STRING;
 Result.argstring := arg;
end;

function TArgRetriever.getFloatArgument(var error : TStkError) : TArgStk;
var arg   : String;
    float : TStkFloat;
begin
 arg := ExtractParam(toparse_, ','); 
 try
   float := StrToFloat(arg);
 except
   Result.argtype := STK_ARG_ERROR;
   error.ErrorID  := COULD_NOT_PARSE_FLOAT_ID;
   error.ErrorMsg := COULD_NOT_PARSE_FLOAT;
   error.ErrorArg := arg;
   Exit;
 end;
 
 Result.argtype  := STK_ARG_FLOAT;
 Result.argvalue := float;
end;

function TArgRetriever.getExpressionArgument(openbracket : String; var error : TStkError) : TArgStk;
begin
  if openbracket='{' then
     Result := getBracketArgument('{', '}', 
               WRONG_NUMBER_OF_BRACKETS_ID, WRONG_NUMBER_OF_BRACKETS+' ->{}', error)
  else 
  if openbracket='(' then
     Result := getBracketArgument('(', ')', 
               WRONG_NUMBER_OF_BRACKETS_ID, WRONG_NUMBER_OF_BRACKETS+' ->()', error);
end;



function TArgRetriever.getOtherArgument(var error : TStkError) : TArgStk;
var arg,
    lowerarg    : TStkString;
    specialType : TStkArgType;
begin
  arg := Trim(ExtractParam(toparse_, ','));
  lowerarg := lowercase(arg);
  if (lowerarg='true') or (lowerarg='false') then
      Result := getBooleanArgument(error, lowerarg)
  else    
  if Assigned(speccommands_) and speccommands_.isSpecialCommand(arg, specialType) then
      Result := getSpecialArgument(error, arg, specialType)
  else
  Result := getCallArgument(error, arg);  
end;

function TArgRetriever.getBooleanArgument(var error : TStkError; arg : TStkString) : TArgStk;
var value : TStkFloat;
begin
 if (arg='true') then value := 1 else value := 0;
 Result.argtype   := STK_ARG_BOOLEAN;
 Result.argvalue  := value;
end;


function TArgRetriever.getSpecialArgument(var error : TStkError; arg : TStkString; specialType : TStkArgType) : TArgStk;
begin
 Result.argtype   := specialType;
 Result.argstring := arg;
end;


function TArgRetriever.getCallArgument(var error : TStkError; arg : TStkString) : TArgStk;
begin
 Result.argtype   := STK_ARG_CALL;
 Result.argstring := arg;
end;


end.
