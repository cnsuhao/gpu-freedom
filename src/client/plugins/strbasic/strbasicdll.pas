unit strbasicdll;


interface

uses SysUtils, stacks;

function description    : TStkString;
function weburl         : TStkString;
function stkversion     : TStkString;

function concat(var stk: TStack): boolean;
function substr(var stk: TStack): boolean;
function copy(var stk: TStack): boolean;
function delete(var stk: TStack): boolean;
function insert(var stk: TStack): boolean;
function length(var stk: TStack): boolean;
function tostr(var stk: TStack): boolean;
function compare(var stk: TStack): boolean;

function cleanstack(var stk: TStack): boolean;
function testexception(var stk: TStack): boolean;

implementation


function description: TStkString;
begin
  Result := 'Strbasic.dll contains basic functions for strings ' +
    'like concat, substr, copy, delete, insert, length, compare, tostr...';
end;

function weburl: TStkString;
begin
  Result := 'http://gpu.sourceforge.net';
end;

function stkversion: TStkString;
begin
  Result := STACK_VERSION;
end;

function concat(var stk: TStack): boolean;
var
  Idx: integer;
  tmp: string;
  l:   integer;
begin
  Result := False;
  Idx    := stk.StIdx; {this should speed up and also made the code more readable}

  {check if enough parameter}
  if Idx < 2 then
    Exit;

  {check that both parameters are strings}
  if not (Stk.Stack[Idx - 1] = INF) then
    Exit;
  if not (Stk.Stack[Idx] = INF) then
    Exit;

  tmp := StrPas(Stk.PCharStack[Idx - 1]) + StrPas(Stk.PCharStack[Idx]);


  Stk.QCharStack[Idx - 1] := Str2PChar(tmp);

  {never forget to set the Idx right at the end}
  stk.StIdx := Idx - 1;
  Result    := True;
end;

function substr(var stk: TStack): boolean;
var
  Idx, position: integer;
begin
  Result := False;
  Idx    := stk.StIdx;
  if Idx < 2 then
    Exit;
  if not (Stk.Stack[Idx - 1] = INF) then
    Exit;
  if not (Stk.Stack[Idx] = INF) then
    Exit;

  position := Pos(StrPas(Stk.PCharStack[Idx - 1]),
    StrPas(Stk.PCharStack[Idx]));

  Stk.Stack[Idx - 1] := position;
  //Stk.PCharStack[Idx-1] := nil;

  stk.StIdx := Idx - 1;
  Result    := True;
end;

function copy(var stk: TStack): boolean;
var
  Idx:    integer;
  tmp, S: string;
begin
  Result := False;
  Idx    := stk.StIdx;
  if Idx < 3 then
    Exit;
  if not (Stk.Stack[Idx - 2] = INF) then
    Exit;
  if (Stk.Stack[Idx - 1] = INF) then
    Exit;
  if (Stk.Stack[Idx] = INF) then
    Exit;

  S   := StrPas(Stk.PCharStack[Idx - 2]);
  tmp := System.Copy(S, Trunc(Stk.Stack[Idx - 1]),
    Trunc(Stk.Stack[Idx]));

  Stk.Stack[Idx - 2]      := INF;
  Stk.QCharStack[Idx - 2] := Str2PChar(tmp);

  stk.StIdx := Idx - 2;
  Result    := True;
end;

function Delete(var stk: TStack): boolean;
var
  Idx:    integer;
  tmp, S: string;
begin
  Result := False;
  Idx    := stk.StIdx;
  if Idx < 3 then
    Exit;
  if not (Stk.Stack[Idx - 2] = INF) then
    Exit;
  if (Stk.Stack[Idx - 1] = INF) then
    Exit;
  if (Stk.Stack[Idx] = INF) then
    Exit;

  S := StrPas(Stk.PCharStack[Idx - 2]);
  System.Delete(S,
    Trunc(Stk.Stack[Idx - 1]),
    Trunc(Stk.Stack[Idx]));

  tmp := S;
  Stk.Stack[Idx - 2] := INF;
  Stk.QCharStack[Idx - 2] := Str2PChar(tmp);

  stk.StIdx := Idx - 2;
  Result    := True;
end;

function insert(var stk: TStack): boolean;
var
  Idx:    integer;
  tmp, S: string;
begin
  Result := False;
  Idx    := stk.StIdx;
  if Idx < 3 then
    Exit;
  if not (Stk.Stack[Idx - 2] = INF) then
    Exit;
  if not (Stk.Stack[Idx - 1] = INF) then
    Exit;
  if (Stk.Stack[Idx] = INF) then
    Exit;

  S := StrPas(Stk.PCharStack[Idx - 1]);
  System.Insert(StrPas(Stk.PCharStack[Idx - 2]),
    S,
    Trunc(Stk.Stack[Idx]));

  tmp := S;
  Stk.Stack[Idx - 2] := INF;
  Stk.QCharStack[Idx - 2] := Str2PChar(tmp);

  stk.StIdx := Idx - 2;
  Result    := True;
end;

function length(var stk: TStack): boolean;
var
  idx: integer;
begin
  Result := False;
  Idx    := stk.StIdx;
  if Idx < 1 then
    Exit;
  if not (Stk.Stack[Idx] = INF) then Exit;

  Stk.Stack[Idx] := System.Length(StrPas(Stk.PCharStack[Idx]));
  //Stk.PCharStack[Idx] := nil;

  Result := True;
end;

function tostr(var stk: TStack): boolean;
var
  idx: integer;
  FormatSet : TFormatSet;
begin
  Result := False;
  Idx    := stk.StIdx;
  if Idx < 1 then Exit;
  if (Stk.Stack[Idx] = INF) then Exit;


  Stk.PCharStack[Idx] := PChar(FloatToStr(Stk.Stack[Idx],FormatSet.fs));
  Stk.Stack[Idx] := INF;

  Result := True;
end;

function compare(var stk: TStack): boolean;
var
  idx: integer;
  str1, str2 : String;
begin
  Result := False;
  Idx    := stk.StIdx;
  if Idx < 2 then Exit;
  if not (Stk.Stack[Idx] = INF) then Exit;
  if not (Stk.Stack[Idx-1] = INF) then Exit;

  str1 := StrPas(Stk.PCharStack[Idx]);
  str2 := StrPas(Stk.PCharStack[Idx-1]);

  if (str1=str2) then Stk.Stack[Idx-1] :=1 else Stk.Stack[Idx-1] := 0;

  Stk.StIdx := Stk.StIdx - 1;

  Result := True;
end;



function cleanstack(var stk: TStack): boolean;
begin
 cleanStk(stk);
end;


function testexception(var stk: TStack): boolean;
begin
 raise Exception.Create('This exception should not block the entire Virtual Machine');
end;



end.
