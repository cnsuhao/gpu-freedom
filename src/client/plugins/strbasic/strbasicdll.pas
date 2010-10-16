unit strbasicdll;

interface

uses SysUtils, stacks, formatsets;

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

function checkStringParams(nbParams : Longint; var stk : TStack) : Boolean;
var types : TStkTypes;
begin
 types[1]:=STRING_STKTYPE;
 types[2]:=STRING_STKTYPE;
 Result  :=typeOfParametersCorrect(nbParams, stk, types);
end;

function retrieveStringParam(var a : TStkString; var stk : TStack) : Boolean;
begin
 Result := checkStringParams(1, stk);
 if Result then popStr(a, stk) else a := '';
end;

function retrieveStringParams(var a : TStkString; var b: TStkString; var stk : TStack) : Boolean;
begin
 Result := checkStringParams(2, stk);
 if Result then
    begin
      popStr(b, stk);
      popStr(a, stk);
    end
 else
    begin
      a := '';
      b := '';
    end;
end;

function concat(var stk: TStack): boolean;
var a, b : TStkString;
begin
  Result := retrieveStringParams(a, b, stk);
  if Result then pushStr(a+b, stk);
end;

function substr(var stk: TStack): boolean;
var a, b : TStkString;
begin
  Result := retrieveStringParams(a, b, stk);
  if Result then pushFloat(Pos(a, b), stk);
end;

function copy(var stk: TStack): boolean;
var a, b  : TStkFloat;
    s     : TStkString;
    types : TStkTypes;
begin
 types[1]:=STRING_STKTYPE;
 types[2]:=FLOAT_STKTYPE;
 types[3]:=FLOAT_STKTYPE;
 Result  :=typeOfParametersCorrect(3, stk, types);
 if Result then
      begin
        popFloat(b, stk);
        popFloat(a, stk);
        popStr(s, stk);
        pushStr(System.Copy(s, trunc(a), trunc(b)), stk);
      end;
end;

function Delete(var stk: TStack): boolean;
var a, b  : TStkFloat;
    s     : TStkString;
    types : TStkTypes;
begin
 types[1]:=STRING_STKTYPE;
 types[2]:=FLOAT_STKTYPE;
 types[3]:=FLOAT_STKTYPE;
 Result  :=typeOfParametersCorrect(3, stk, types);
 if Result then
      begin
        popFloat(b, stk);
        popFloat(a, stk);
        popStr(s, stk);
        System.Delete(s, trunc(a), trunc(b));
        pushStr(s, stk);
      end;
end;

function insert(var stk: TStack): boolean;
var s1, s2 : TStkString;
    a      : TStkFloat;
    types  : TStkTypes;
begin
 types[1]:=STRING_STKTYPE;
 types[2]:=STRING_STKTYPE;
 types[3]:=FLOAT_STKTYPE;
 Result  :=typeOfParametersCorrect(3, stk, types);
 if Result then
      begin
        popFloat(a, stk);
        popStr(s2, stk);
        popStr(s1, stk);
        System.Insert(s1, s2, trunc(a));
        pushStr(s2, stk);
      end;

end;

function length(var stk: TStack): boolean;
var s : TStkString;
begin
  Result := retrieveStringParam(s, stk);
  if Result then pushFloat(trunc(System.length(s)), stk);
end;

function tostr(var stk: TStack): boolean;
var a : TStkFloat;
begin
  Result := popFloat(a, stk);
  if Result then pushStr(FloatToStr(a), stk);
end;

function compare(var stk: TStack): boolean;
var str1, str2 : TStkString;
begin
 Result := retrieveStringParams(str1, str2, stk);
 if result then
     pushBool(str1=str2, stk);
end;



function cleanstack(var stk: TStack): boolean;
begin
 clearStk(stk);
end;


function testexception(var stk: TStack): boolean;
begin
 Result := false;
 raise Exception.Create('This exception should not block the entire Virtual Machine');
end;



end.
