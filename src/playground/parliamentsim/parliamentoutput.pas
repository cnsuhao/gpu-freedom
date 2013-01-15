unit parliamentoutput;

{$mode objfpc}{$H+}

interface

uses
  Classes, SysUtils, parliamentstructure;

procedure log(str, filename : AnsiString);
procedure printParliament;

implementation

procedure log(str, filename : AnsiString);
var F : Textfile;
begin
  WriteLn(str);
  if not FileExists(filename) then
     begin
       AssignFile(F, filename);
       Rewrite(F);
       CloseFile(F);
     end;

  AssignFile(F, filename);
  Append(F);
  WriteLn(F, str);
  CloseFile(F);
end;


procedure printDelegate(i : Longint);
begin
  log(IntToStr(i)+';'+FloatToStr(parliament.delegates.delg[i].personalinterest)+';'+
                      FloatToStr(parliament.delegates.delg[i].collectiveinterest)+';'+
                      IntToStr(parliament.delegates.delg[i].party), 'parliament.txt');
end;

procedure printParliament;
var i :  Longint;
begin
  log('Delegates:', 'parliament.txt');
  for i:=1 to parliament.delegates.size do
      printDelegate(i);
end;


end.

