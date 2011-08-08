unit anagramsdll;

interface

uses SysUtils, stacks, formatsets;

function description    : TStkString;
function weburl         : TStkString;
function stkversion     : TStkString;

function anagram(var stk: TStack): boolean;
function twowordsanagram(var stk: TStack): boolean;

implementation


function description: TStkString;
begin
  Result := 'Anagrams.dll is able to produce anagrams for single words and for two words.';
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

function anagram(var stk: TStack): boolean;
var a, b : TStkString;
begin
  Result := retrieveStringParams(a, b, stk);
  if Result then pushStr(a+b, stk);
end;

function twowordsanagram(var stk: TStack): boolean;
var a, b : TStkString;
begin
  Result := retrieveStringParams(a, b, stk);
  if Result then pushFloat(Pos(a, b), stk);
end;


end.
