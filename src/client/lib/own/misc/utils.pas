unit utils;

interface

uses SysUtils;

function createUniqueId() : String;
function ExtractParam(var S: string; Separator: string): string;
function ExtractParamLong(var S: AnsiString; Separator: string): AnsiString;


implementation

function createUniqueId() : String;
var outG : TGuid;
begin
  createGUID(outG);
  Result := GUIDToString(outG);
end;

function ExtractParam(var S: string; Separator: string): string;
var
  i: Longint;
begin
  i := Pos(Separator, S);
  if i > 0 then
  begin
    Result := Copy(S, 1, i - 1);
    Delete(S, 1, i-1);
    Delete(S, 1, length(Separator));
  end
  else
  begin
    Result := S;
    S      := '';
  end;
end;


function ExtractParamLong(var S: AnsiString; Separator: string): AnsiString;
var
  i: Longint;
begin
  i := Pos(Separator, S);
  if i > 0 then
  begin
    Result := Copy(S, 1, i - 1);
    Delete(S, 1, i-1);
    Delete(S, 1, length(Separator));
  end
  else
  begin
    Result := S;
    S      := '';
  end;
end;

end.

