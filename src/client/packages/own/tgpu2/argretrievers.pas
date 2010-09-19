unit argretrievers;

interface

uses SysUtils, stacks, plugins, utils, common;

const
  GPU_CALL       = 10;
  GPU_STRING     = 20;
  GPU_FLOAT      = 30;
  GPU_EXPRESSION = 40;
  GPU_ERROR      = 99;

type TArgGPU = record
     argtype : ShortInt;
     argvalue  : TGPUFloat;
     argstring : String;
end;

type TArgRetriever = class(TObject);
 public 
   constructor Create(job : String);
   function getJob() : String;
   
   function hasArguments() : Boolean;
   function getArgument(var error : TGPUError) : TArgGPU;
   
 private
   job_,
   toparse_   : String;
   
   procedure deleteComma; 
   function getBracketArgument(openbracket, closebracket : String; 
                               errorID : Longint; errorMsg : String; 
                               var error : TGPUError) : TArgGPU;
   function getStringArgument(var error : TGPUError) : TArgGPU;
   function getFloatArgument(var error : TGPUError) : TArgGPU;
   function getCallArgument(var error : TGPUError) : TArgGPU;
   function getExpressionArgument(openbracket : String; var error : TGPUError) : TArgGPU;

end;


implementation

constructor TArgRetriever.Create(job : String);
begin
 job_ := Trim(job);
 toparse_ := job_;
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
       Result.argtype := GPU_ERROR;
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
 if (ordchar>=48 and ordchar<=57) or startchar='.'
     Result := getFloatArgument(error)
 else
 if (startchar = '{') or (startchar = '(')  then
     Result := getExpressionArgument(startchar, error)
 else    
 // it has to be a method call inside a plugin
     Result := getCallArgument(error);
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

    while (bracketCount <> 0) and (i <= Length(S)) do
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
      Result.argtype := GPU_EXPRESSION;
      Result.argstring := arg;
    end
    else {problem in brackets}
    begin
      Result.argtype   := GPU_ERROR;
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

 if (i=Length(toparse_) and toparse_[i]<>QUOTE then
     begin
       // quote is not closed
       Result.argtype := GPU_ERROR;
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
 
 Result.argtype   := GPU_STRING;
 Result.argstring := arg;
end;

function TArgRetriever.getFloatArgument(var error : TGPUError) : TArgGPU;
var arg : String;
    float : TGPUFloat;
begin
 arg := ExtractParam(toparse_, ','); 
 try
   float := StrToFloat(arg, formatSet.fs);
 except
   Result.argtype := GPU_ERROR;
   error.ErrorID  := COULD_NOT_PARSE_FLOAT_ID;
   error.ErrorMsg := COULD_NOT_PARSE_FLOAT;
   error.ErrorArg := arg;
   Exit;
 end;
 
 Result.argtype  := GPU_FLOAT;
 Result.argvalue := float;
end;

function TArgRetriever.getCallArgument(var error : TGPUError) : TArgGPU;
var arg : String;
begin
 arg := ExtractParam(toparse_, ',');
 Result.argtype   := GPU_CALL;
 Result.argstring := arg;
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





end.