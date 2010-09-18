unit gpu_utils;
{
This unit contains help functions for the ComputationThread.
See also http://gpu.sourceforge.net/virtual.php
}

interface

uses
  SysUtils, definitions;

function isFloat(S: string): boolean;
function ReturnArg(var S: string): string;
function NumberInSquareBrackets(S: string): integer;
function ExtractParam(var S: string; Separator: string): string;

 //allocates mem for PChar
 //and copies it
function StringToPChar(S: string): PChar;

implementation

function isFloat(S: string): boolean;
var
  number: real;
  code:   integer;
begin
  Val(S, number, code);
  Result := (code = 0);
end;

{return argument from beginning of string to position with next comma,
 the argument is deleted from the S string passed by reference.}
function ReturnArg(var S: string): string;
var
  iPos, bracketCount, i: integer;

begin
  TrimLeft(S);
  {this is the special GPU string to be loaded in the PChar stack}
  if Pos(APOSTROPHE, S) = 1 then
  begin
    i := 2;
    while (S[i] <> APOSTROPHE) and (i <= Length(S)) do
      Inc(i);

    Result := Copy(S, 1, i);
    Delete(S, 1, i);
    {delete an eventual comma}
    TrimLeft(S);
    if Pos(',', S) = 1 then
      Delete(S, 1, 1);
    Exit;
  end;

  if Pos('{', S) = 1 then
  begin
                {string begins with bracket, we will return an argument
                 enclosed between two brackets.}
    bracketCount := 1;
    i := 2;

    while (bracketCount <> 0) and (i <= Length(S)) do
    begin
      if S[i] = '{' then
        Inc(BracketCount)
      else
      if S[i] = '}' then
        Dec(BracketCount);
      Inc(i);
    end;

    if bracketCount = 0 then
    begin
      Result := Copy(S, 1, i - 1);
                        {i-1 because we increment when we find
                         the right closing bracket}
      Delete(S, 1, i - 1);
      TrimLeft(S);
      if Pos(',', S) = 1 then
        Delete(S, 1, 1);
    end
    else {problem in brackets}
    begin
      Result := '';
      S      := '';
    end;
    Exit;
  end;

  iPos := Pos(',', S);
  if (iPos = 0) then
  begin
    {String comma was not found}
    Result := S;
    S      := '';
  end
  else
  begin
    Result := Copy(S, 1, iPos - 1);
    {-1 because comma is not an argument character}
    Delete(S, 1, iPos);
  end;
end; {ReturnArg}


function NumberInSquareBrackets(S: string): integer;
var
  iPos, ResultInt: integer;
  StrNum: string;
begin
  iPos := Pos('[', S);
  if iPos = 1 then
  begin
    Delete(S, 1, 1);
    iPos := Pos(']', S);
    if iPos <> 0 then
    begin
      StrNum := Copy(S, 1, iPos - 1);
      try
        ResultInt := StrToInt(StrNum);
      except
        ResultInt := 0;
      end;{try}
    end
    else
      ResultInt := 0;

  end
  else
    ResultInt := 0;

  if (ResultInt < 1) or (ResultInt > MAX_COLLECTING_IDS) then
    ResultInt := 0;
  Result := ResultInt;
end; {NumberInSquareBrackets}

function StringToPChar(S: string): PChar;
begin
  if s = '' then
  begin
    Result := nil;
    exit;
  end;
  GetMem(Result, length(S) + 1);
  StrPCopy(Result, S);
end;


function ExtractParam(var S: string; Separator: string): string;
var
  i: Longint;
begin
  i := Pos(Separator, S);
  if i > 0 then
  begin
    Result := Copy(S, 1, i - 1);
    Delete(S, 1, i);
  end
  else
  begin
    Result := S;
    S      := '';
  end;
end;


end.
