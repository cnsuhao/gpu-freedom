unit anagramsdll;

interface

uses SysUtils, stacks, formatsets, anagramsunit;

function description    : TStkString;
function weburl         : TStkString;
function stkversion     : TStkString;

function discard(var stk: TStack): boolean;
function anagram(var stk: TStack): boolean;
function twowordsanagram(var stk: TStack): boolean;

implementation

var wExistence       : TWordExistence;
    wexistenceLoaded : Boolean;

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

procedure loadWordExistence;
var appPath : String;
begin
 if not wExistenceLoaded then
     begin
      appPath := ExtractFilePath(ParamStr(0));
      wExistence := TWordExistence.Create(appPath+PathDelim+'wordlist.txt');
      wExistenceLoaded := true;
     end;
end;

function discard(var stk: TStack): boolean;
begin
 if wExistenceLoaded then wExistence.Free;
end;

function anagram(var stk: TStack): boolean;
var a, res    : TStkString;
    myAnagram : TAnagram;
begin
  loadWordExistence;
  Result := retrieveStringParam(a, stk);
  if Result then
    begin
     myAnagram := TAnagram.Create(a, wExistence);
     res := myAnagram.findAnagram;
     pushStr(res, stk);
     myAnagram.Free;
     Result := res<>'';
    end;
end;

function twowordsanagram(var stk: TStack): boolean;
var a,res   : TStkString;
    myTwoAnagram : TTwoWordAnagram;
begin
  loadWordExistence;
  Result := retrieveStringParam(a, stk);
  if Result then
    begin
     myTwoAnagram := TTwoWordAnagram.Create(a, wExistence);
     res := myTwoAnagram.findTwoWordsAnagram;
     pushStr(res, stk);
     myTwoAnagram.Free;
     Result := res<>'';
    end;

end;


end.
