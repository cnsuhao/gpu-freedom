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

type TArgGPU = record
     argtype : ShortInt;
     argvalue  : TGPUFloat;
     argstring : String;
end;

type TArgRetriever = class(TObject)
 public 
   constructor Create(job : String; var speccommands : TSpecialCommand);
   function getJob() : String;
   
   function hasArguments() : Boolean;
   function getArgument(var error : TGPUError) : TArgGPU;
   
 private
   job_,
   toparse_      : String;
   speccommands_ : TSpecialCommand;
   
   procedure deleteComma; 
   function getBracketArgument(openbracket, closebracket : String; 
                               errorID : Longint; errorMsg : String; 
                               var error : TGPUError) : TArgGPU;
   function getStringArgument(var error : TGPUError) : TArgGPU;
   function getFloatArgument(var error : TGPUError) : TArgGPU;
   function getExpressionArgument(openbracket : String; var error : TGPUError) : TArgGPU;

   function getOtherArgument(var error : TGPUError) : TArgGPU;
   function getSpecialArgument(var error : TGPUError; arg : String; specialType : Longint) : TArgGPU;
   function getBooleanArgument(var error : TGPUError;arg : String) : TArgGPU;
   function getCallArgument(var error : TGPUError; arg : String) : TArgGPU;
   
end;


implementation

constructor TArgRetriever.Create(job : String; var speccommands : TSpecialCommand);
begin
 job_ := Trim(job);
 toparse_ := job_;
 speccommands_ := speccommands;
end;

function TArgRetriever.getJob() : String;
begin
  Result := job_;
end;

function TArgRetriever.hasArguments() : Boolean;
begin
  Result := toparse_<> '';
end;

function TArgRetriever.getArgument(var error : TGPUError) : TArgGPU;
var startchar : Char;
    ordchar   : ShortInt;
begin
 if toparse_= '' then raise Exception.Create('getArgument() called when there are no arguments!');
 
 startchar := toparse_[1];
 ordchar := Ord(startchar);
 if startchar = ',' then
    begin
       // an empty argument, raise error
       Result.argtype := GPU_ARG_ERROR;
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
 if ((ordchar>=48) and (ordchar<=57)) or (startchar='.') then
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
 TrimLeft(toparse_);
 if Pos(',', toparse_) = 1 then
      Delete(toparse_, 1, 1);
end;

function TArgRetriever.getBracketArgument(openbracket, closebracket : String; errorID : Longint; errorMsg : String; 
                            var error : TGPUError) : TArgGPU;
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
      arg := Copy(toparse_, 2, i - 2);
      // i-1 because we increment when we find
      // the right closing bracket
      Delete(toparse_, 1, i - 1);
      deleteComma();
      Result.argtype := GPU_ARG_EXPRESSION;
      Result.argstring := arg;
    end
    else {problem in brackets}
    begin
      Result.argtype   := GPU_ARG_ERROR;
      error.errorID := errorID;
      error.errorMsg := errorMsg;
      error.errorArg := toparse_;
    end;
end;

function TArgRetriever.getStringArgument(var error : TGPUError) : TArgGPU;
var i : Longint;
    arg : String;
begin
 i := 2;
 while (toparse_[i] <> QUOTE) and (i <= Length(toparse_)) do
      Inc(i);

 if (i=Length(toparse_)) and (toparse_[i]<>QUOTE) then
     begin
       // quote is not closed
       Result.argtype := GPU_ARG_ERROR;
       error.errorID := MISSING_QUOTE_ID;
       error.errorMsg := MISSING_QUOTE;
       error.errorArg := toparse_;
       Exit;
     end;     
 
 if (i=2) then
   arg := '' // empty string
 else   
   arg := Copy(toparse_, 2, i-1);
 
 Delete(toparse_, 1, i);
 deleteComma();
 
 Result.argtype   := GPU_ARG_STRING;
 Result.argstring := arg;
end;

function TArgRetriever.getFloatArgument(var error : TGPUError) : TArgGPU;
var arg : String;
    float : TGPUFloat;
begin
 arg := ExtractParam(toparse_, ','); 
 try
   float := StrToFloat(arg);
 except
   Result.argtype := GPU_ARG_ERROR;
   error.ErrorID  := COULD_NOT_PARSE_FLOAT_ID;
   error.ErrorMsg := COULD_NOT_PARSE_FLOAT;
   error.ErrorArg := arg;
   Exit;
 end;
 
 Result.argtype  := GPU_ARG_FLOAT;
 Result.argvalue := float;
end;

function TArgRetriever.getExpressionArgument(openbracket : String; var error : TGPUError) : TArgGPU;
begin
  if openbracket='{' then
     Result := getBracketArgument('{', '}', 
               WRONG_NUMBER_OF_BRACKETS_ID, WRONG_NUMBER_OF_BRACKETS+' ->{}', error)
  else 
  if openbracket='(' then
     Result := getBracketArgument('(', ')', 
               WRONG_NUMBER_OF_BRACKETS_ID, WRONG_NUMBER_OF_BRACKETS+' ->()', error);
end;



function TArgRetriever.getOtherArgument(var error : TGPUError) : TArgGPU;
var arg,
    lowerarg : String;
	specialType : Longint;
begin
  arg := Trim(ExtractParam(toparse_, ','));
  lowerarg := lowercase(arg);
  if (lowerarg='true') or (lowerarg='false') then
      Result := getBooleanArgument(error, lowerarg)
  else    
  if speccommands_.isSpecialCommand(arg, specialType) then
      Result := getSpecialArgument(error, arg, specialType)
  else
  Result := getCallArgument(error, arg);  
end;

function TArgRetriever.getBooleanArgument(var error : TGPUError; arg : String) : TArgGPU;
var value : TGPUFloat;
begin
 if (arg='true') then value := 1 else value := 0;
 Result.argtype   := GPU_ARG_BOOLEAN;
 Result.argvalue  := value;
end;


function TArgRetriever.getSpecialArgument(var error : TGPUError; arg : String; specialType : Longint) : TArgGPU;
begin
 Result.argtype   := specialType;
 Result.argstring := arg;
end;


function TArgRetriever.getCallArgument(var error : TGPUError; arg : String) : TArgGPU;
begin
 Result.argtype   := GPU_ARG_CALL;
 Result.argstring := arg;
end;


end.
