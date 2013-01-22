unit parliamentoutput;

{$mode objfpc}{$H+}

interface

uses
  Classes, SysUtils, parliamentstructure;

procedure log(str, filename : AnsiString);
procedure printParliament(var parliament : TParliament);
procedure printStatistic(var stats : TSimstats; round : Longint);

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


procedure printDelegate(var parliament : TParliament; i : Longint);
begin
  log(IntToStr(i)+';'+FloatToStr(parliament.delegates.delg[i].personalinterestx)+';'+
                      FloatToStr(parliament.delegates.delg[i].collectiveinteresty)+';'+
                      IntToStr(parliament.delegates.delg[i].party)
                      , 'delegates.csv');
end;

procedure printParty(var parliament : TParliament; i : Longint);
begin
  log(IntToStr(i)+';'+FloatToStr(parliament.parties.par[i].centerx)+';'+
                      FloatToStr(parliament.parties.par[i].centery)+';'+
                      FloatToStr(parliament.parties.par[i].radius)+';'+
                      IntToStr(parliament.parties.par[i].size), 'parties.csv');
end;

procedure printParliament(var parliament : TParliament);
var i :  Longint;
begin
  {
  log('Delegates:', 'parliament.txt');
  for i:=1 to parliament.delegates.size do
      printDelegate(parliament, i);
  }
  log('', 'parliament.txt');
  log('Parties:', 'parliament.txt');
  for i:=1 to parliament.parties.size do
      printParty(parliament, i);

  WriteLn;
  WriteLn('Indipendents: '+IntToStr(parliament.indipendents));
end;

procedure printStatistic(var stats : TSimstats; round : Longint);
begin
   log(IntToStr(round)+';'+IntToStr(stats.legislatures[round].nbdelegates)+';'+
                      IntToStr(stats.legislatures[round].nbindipendents)+';'+
                      //IntToStr(stats.legislatures[round].nbparties)+';'+
                      IntToStr(stats.legislatures[round].nbparty1)+';'+
                      IntToStr(stats.legislatures[round].nbparty2)+';'+
                      FloatToStr(stats.legislatures[round].approvalrate)+';'+
                      FloatToStr(stats.legislatures[round].totalbenefit)+';', 'statistics.csv');
end;

end.

