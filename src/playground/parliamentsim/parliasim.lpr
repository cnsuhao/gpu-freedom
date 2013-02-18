program parliasim;

{$mode objfpc}{$H+}

uses
  {$IFDEF UNIX}{$IFDEF UseCThreads}
  cthreads,
  {$ENDIF}{$ENDIF}
  Classes, SysUtils, CustApp, parliamentstructure, parliamentoutput, simulation,
  statistics, analysis
  { you can add units after this };

const
      PARLIAMENT_SIZE = 300;    // number of delegates in parliament
      PARLIAMENT_ROUNDS = 1000; // how many rounds for each majorPartyPercentage sweep over legislatures
      LAWS_PROPOSALS = 365;     // how many laws proposals in a legislature are proposed (motions)

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
var   par         : TParliament;
      laws        : TLaws;
      simstats    : TSimStats;
      bigpicstats : TBigPictureStats;

procedure simulateLegislature(round, parliamentsize, nbindipendents : Longint; majorPartyPercentage : Extended);
var count : Longint;
begin
  count := 0;
  while (not initParliamentv2(par, parliamentsize, nbindipendents, 0.5 {partyradius}, majorPartyPercentage))  do
        begin
          Inc(count);
          if count>1000 then raise Exception.Create('Unable to start legislature!');
        end;
  //WriteLn('Parliament constituted after '+IntToStr(count)+' tries...');

  //printParliament(par);
  initLaws(laws, LAWS_PROPOSALS);
  simulateParliament(par, laws);
  collectStatistics(par, laws, simstats, round);
  //printStatistic(simstats, round);
end;


procedure simulateLegislatures(biground, parliamentsize, nbindipendents : Longint; majorPartyPercentage : Extended);
var i : Longint;
begin
  initSimStats(simstats, PARLIAMENT_ROUNDS);
  for i:=1 to PARLIAMENT_ROUNDS do
    simulateLegislature(i, parliamentsize, nbindipendents, majorPartyPercentage);

  collectBigPictureStats(biground, parliamentsize, nbindipendents, majorPartyPercentage, simstats, bigpicstats);
  printBigPictureStats(biground, bigpicstats);
end;



procedure searchOptimalNumberOfIndipendents(parliamentsize : Longint; majorPartyPercentage : Extended);
var i : Longint;
begin
  WriteLn('Optimal number of indipendents should be: '+
          IntToStr( optimalIndipendentsNumber(parliamentsize, majorPartyPercentage) ));

  for i:=0 to parliamentsize do
     simulateLegislatures(i, parliamentsize, i, majorPartyPercentage);

  WriteLn('Finishing this round with following results: ');
  printOptimalIndipendentRate(bigpicstats, parliamentsize, majorPartyPercentage);
end;

procedure TParliamentSim.DoRun;
var tick, majorPartyPercentage : Extended;
    i    : Longint;
begin
  tick := 0.05;
  for i:=0 to 9 do
     begin
       majorPartyPercentage := 0.5 + i * tick;
       WriteLn;
       WriteLn('Searching parameter space with '+IntToStr(PARLIAMENT_SIZE)+
               ' delegates and percentage of major party '+
               FloatToStr(majorPartyPercentage));

       searchOptimalNumberOfIndipendents(PARLIAMENT_SIZE, majorPartyPercentage);

       WriteLn;
     end;
  WriteLn('Parliament simulation finished, check optimal-vs-simulation.csv for results.');
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

