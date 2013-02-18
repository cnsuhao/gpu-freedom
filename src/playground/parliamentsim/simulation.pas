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


function getPersonalInterest(var parliament : TParliament; del : Longint) : Extended;
var party : Longint;
begin
  party := parliament.delegates.delg[del].party;
  if  party = INDIPENDENT then
     Result := parliament.delegates.delg[del].personalinterestx
  else
     Result := parliament.parties.par[party].centerx;
end;

function getCollectiveInterest(var parliament : TParliament; del : Longint) : Extended;
var party : Longint;
begin
  party := parliament.delegates.delg[del].party;
  if  party = INDIPENDENT then
     Result := parliament.delegates.delg[del].collectiveinteresty
  else
     Result := parliament.parties.par[party].centery;
end;

// defines the voting window
function delegateAcceptsLaw(var parliament : TParliament; var laws : TLaws; l, del : Longint) : Boolean;
var x, y : Extended;
begin
   x := getPersonalInterest(parliament, del);
   y := getCollectiveInterest(parliament, del);

   Result := (laws.laws[l].personalinterestx>=x) and (laws.laws[l].collectiveinteresty>=y);
end;

procedure voteLaw(var parliament : TParliament; var laws : TLaws; l : Longint);
var i : Longint;
begin
  for i:=1 to parliament.delegates.size do
       begin
         if delegateAcceptsLaw(parliament, laws, l, i) then
            begin
              Inc(laws.laws[l].yes);
            end
          else
            begin
              Inc(laws.laws[l].no);
            end;

       end;

  laws.laws[l].approved := laws.laws[l].yes > laws.laws[l].no;
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

