unit parliamentstructure;

{$mode objfpc}{$H+}

interface

uses
  Classes, SysUtils, isaac;

const MAX_DELEGATES = 300;
      MAX_PARTIES = 5;
      INDIPENDENT = -1;
      MAX_TRIES = 10000;

type TDelegate = record
    personalinterest,
    collectiveinterest : Extended; // values between -1..1

    party : Longint; // index to the party to which the delegate belongs
                     // -1 if INDIPENDENT
end;

type TDelegates = record
    delg : Array[1..MAX_DELEGATES] of TDelegate;
    size : Longint;
end;

type TParty  = record
    size    : Longint;

    centerx,
    centery,
    radius   : Extended;
end;

type TParties = record
    par : Array [1..MAX_PARTIES] of TParty;
    size : Longint;
end;

type TParliament = record
    delegates   : TDelegates;
    parties : TParties;

    indipendents : Longint; // number of delegates not covered by a party
end;

var rndgen     : TIsaac;
    parliament : TParliament;

function getRndminusOnetoOne : Extended;
function distance(x,y,x2,y2 : Extended) : Extended;

procedure initParliament(nbdelegates, nbparties : Longint; partyradius : Extended);

implementation

function getRndminusOnetoOne : Extended;
begin
  Result := ( ( rndgen.Val/High(cardinal) )  * 2) - 1;
end;

function distance(x,y,x2,y2 : Extended) : Extended;
begin
  Result := Sqrt((x2-x)*(x2-x)+(y2-y)*(y2-y));
end;

// this garuantess a center with a complete circle
function getPartyCenter(partyradius : Extended) : Extended;
begin
  Result := ( ( rndgen.Val/High(cardinal) )  * (2-2*partyradius)  ) - 1 + partyradius;
end;

procedure initDelegate(i : Longint);
begin
  parliament.delegates.delg[i].personalinterest := getRndminusOnetoOne;
  parliament.delegates.delg[i].collectiveinterest := getRndminusOnetoOne;

  parliament.delegates.delg[i].party := INDIPENDENT;
end;


// this garuantees that there is no overlapping between parties
function checkParty(x,y : Extended; party : Longint) : Boolean;
var i : Longint;
begin
  Result := true;
  for i:=1 to party-1 do
       begin
          if distance(x,y,parliament.parties.par[i].centerx,parliament.parties.par[i].centery)<
                      parliament.parties.par[i].radius then
                        begin
                          Result := false;
                          Exit;
                        end;
       end;
end;


procedure initParty(i : Longint; partyradius : Extended);
var x, y : Extended;
    found : Boolean;
    count : Longint;
begin
  parliament.parties.par[i].radius:= partyradius;

  count := 0;
  repeat
   x := getPartyCenter(partyradius);
   y := getPartyCenter(partyradius);
   found := checkParty(x,y, i);

   Inc(count);
   if count>MAX_TRIES then raise Exception.Create('Unable to allocate parties! Try to reduce their number or their size!');
  until found;
  parliament.parties.par[i].centerx:= x;
  parliament.parties.par[i].centery:= y;

  parliament.parties.par[i].size := 0;
end;

procedure assignDelegatesWithPartyRadius(party : Longint);
var x, y : Extended;
begin
   x := parliament.parties.par[party].centerx;
   y := parliament.parties.par[party].centery;

end;

procedure initParliament(nbdelegates, nbparties : Longint; partyradius : Extended);
var i : Longint;
begin
 if nbdelegates>MAX_DELEGATES then raise Exception.Create('Too many delegetes!');

 parliament.delegates.size:=nbdelegates;
 parliament.parties.size := nbparties;

 for i:=1 to nbdelegates do
     begin
        initDelegate(i);
     end;

 // we define the parties and their radius
 for i:=1 to parliament.parties.size do
     begin
        initParty(i, partyradius);
     end;

 // we have two rounds to assign delegates to parties, first we use the setting partyradius
 for i:=1 to parliament.parties.size do
     begin
        assignDelegatesWithPartyRadius(i);
     end;

 // to reach the proposed quota of indipendents, we take delegates and assign them to the party
end;

initialization
  rndgen := TIsaac.Create;

end.
