program parliasim;

{$mode objfpc}{$H+}

uses
  {$IFDEF UNIX}{$IFDEF UseCThreads}
  cthreads,
  {$ENDIF}{$ENDIF}
  Classes, SysUtils, CustApp, parliamentstructure, parliamentoutput, simulation,
  statistics, analysis
  { you can add units after this };

const PARLIAMENT_ROUNDS = 100;
      LAWS_PROPOSALS = 3000;

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


procedure simulateLegislature(round, parliamentsize, nbindipendents : Longint; majorPartyPercentage : Extended);
var count : Longint;
begin
  count := 0;
  while (not initParliamentv2(par, parliamentsize, nbindipendents, 0.5 {partyradius}, majorPartyPercentage))  do
        begin
          Inc(count);
          if count>1000 then raise Exception.Create('Unable to start legislature!');
        end;
  WriteLn('Parliament constituted after '+IntToStr(count)+' tries...');

  //printParliament(par);
  initLaws(laws, LAWS_PROPOSALS);
  simulateParliament(par, laws);
  collectStatistics(par, laws, simstats, round);
  printStatistic(simstats, round);
end;


procedure simulateLegislatures(biground, parliamentsize, nbindipendents : Longint; majorPartyPercentage : Extended);
var i : Longint;
begin
  initSimStats(simstats, PARLIAMENT_ROUNDS);
  for i:=1 to PARLIAMENT_ROUNDS do
    simulateLegislature(i, parliamentsize, nbindipendents, majorPartyPercentage);

  //collectBigPictureStats(biground, parliamentsize, nbindipendents, majorPartyPercentage, simstats);
end;

procedure TParliamentSim.DoRun;
var ErrorMsg: String;
begin
  simulateLegislatures(1, 300, 30, 0.7);
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

