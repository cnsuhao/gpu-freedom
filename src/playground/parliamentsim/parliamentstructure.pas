unit parliamentstructure;

{$mode objfpc}{$H+}

interface

uses
  Classes, SysUtils, isaac;

const MAX_DELEGATES = 300;
      MAX_PARTIES = 5;
      INDIPENDENT = -1;

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
end;

var rndgen     : TIsaac;
    parliament : TParliament;

function getRndminusOnetoOne : Extended;
procedure initParliament(nbdelegates, nbindipendents, nbparties : Longint; partyradius : Extended);

implementation

function getRndminusOnetoOne : Extended;
begin
  Result := ( ( rndgen.Val/High(cardinal) )  * 2) - 1;
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

procedure initParty(i : Longint; partyradius : Extended);
begin
  parliament.parties.par[i].centerx:= getPartyCenter(partyradius);
  parliament.parties.par[i].centery:= getPartyCenter(partyradius);

  parliament.parties.par[i].size := 0;
end;

procedure assignDelegatesWithPartyRadius(party : Longint);
var x, y : Extended;
begin
   x := parliament.parties.par[party].centerx;
   y := parliament.parties.par[party].centery;

end;

procedure initParliament(nbdelegates, nbindipendents, nbparties : Longint; partyradius : Extended);
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

