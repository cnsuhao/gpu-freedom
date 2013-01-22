program parliasim;

{$mode objfpc}{$H+}

uses
  {$IFDEF UNIX}{$IFDEF UseCThreads}
  cthreads,
  {$ENDIF}{$ENDIF}
  Classes, SysUtils, CustApp, parliamentstructure, parliamentoutput, simulation,
  statistics, analysis
  { you can add units after this };

const ROUNDS = 100;

type

  { TParliamentSim }

  TParliamentSim = class(TCustomApplication)
  protected
    procedure DoRun; override;
  public
    constructor Create(TheOwner: TComponent); override;
    destructor Destroy; override;
  end;

{ TParliamentSim }
var   par      : TParliament;
      laws     : TLaws;
      simstats : TSimStats;


procedure simulateLegislature(round : Longint);
var count : Longint;
begin
  count := 0;
  while (not initParliamentv2(par, 300, 30, 0.5, 0.7))  do
        begin
          Inc(count);
          if count>1000 then raise Exception.Create('Unable to start legislature!');
        end;
  WriteLn('Parliament constituted after '+IntToStr(count)+' tries...');

  //printParliament(par);
  initLaws(laws, 3000);
  initSimStats(simstats, 1000);
  simulateParliament(par, laws);
  collectStatistics(par, laws, simstats, round);
  printStatistic(simstats, round);
end;

procedure TParliamentSim.DoRun;
var
  ErrorMsg: String;
  var i : Longint;
begin
  for i:=1 to ROUNDS do
    simulateLegislature(i);

  WriteLn('Parliament simulation finished');
  Readln;
  Terminate;
end;

constructor TParliamentSim.Create(TheOwner: TComponent);
begin
  inherited Create(TheOwner);
end;

destructor TParliamentSim.Destroy;
begin
  inherited Destroy;
end;

var
  Application: TParliamentSim;
begin
  Application:=TParliamentSim.Create(nil);
  Application.Title:='ParliamentSim';
  Application.Run;
  Application.Free;
end.

