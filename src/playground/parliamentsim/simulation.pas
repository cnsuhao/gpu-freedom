unit simulation;

{$mode objfpc}{$H+}

interface

uses
  Classes, SysUtils, parliamentstructure;

procedure simulateParliament(var parliament : TParliament; var laws : TLaws);

implementation

function selectRandomDelegate(var parliament : TParliament) : Longint;
begin
  Result := Trunc(((rndgen.Val/High(cardinal) * (parliament.delegates.size-1))))+1;
end;

procedure createLaw(var parliament : TParliament; var laws : TLaws; l : Longint);
var proposer : Longint;
begin
  proposer := selectRandomDelegate(parliament);
  laws.laws[l].personalinterestx:=parliament.delegates.delg[proposer].personalinterestx;
  laws.laws[l].collectiveinteresty:=parliament.delegates.delg[proposer].collectiveinteresty;

  laws.laws[l].yes := 0;
  laws.laws[l].no  := 0;
end;


procedure simulateLawProcess(var parliament : TParliament; var laws : TLaws; l : Longint);
begin
   createLaw(parliament, laws, l);
   voteLaw(parliament, laws, l);
end;

procedure simulateParliament(var parliament : TParliament; var laws : TLaws);
var l : Longint;
begin
  for l:=1 to laws.size do
    simulateLawProcess(parliament, laws, l);
end;

end.

