unit parliamentstructure;

{$mode objfpc}{$H+}

interface

uses
  Classes, SysUtils, isaac;

const MAX_DELEGATES = 300;
      MAX_PARTIES = 5;
      INDIPENDENT = -1;
      MAX_TRIES = 10000;
      MAX_LAWS = 3650;
      MAX_LEGISLATURES = 3000;

type TDelegate = record
    personalinterestx,
    collectiveinteresty : Extended; // values between -1..1

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

type TLaw = record
    personalinterestx,
    collectiveinteresty : Extended; // values between -1..1

    yes, no  : Longint;
    approved : Boolean;
end;

type TLaws = record
      laws : Array [1..MAX_LAWS] of TLaw;
      size : Longint;

      approved : Longint;
end;

type TLegislatureStats = record
    nbdelegates,
    nbparties,
    nbindipendents : Longint;

    approvalrate,
    totalbenefit : Extended;
end;

type TSimStats = record
    legislatures : Array[1..MAX_LEGISLATURES] of TLegislatureStats;
    size : Longint;
end;


var rndgen     : TIsaac;

function getRndminusOnetoOne : Extended;
function distance(x,y,x2,y2 : Extended) : Extended;

function initParliament(var parliament : TParliament; nbdelegates, nbparties : Longint; partyradius : Extended) : Boolean;
function initParliamentv2(var parliament : TParliament; nbdelegates, nbindipendents : Longint; partyradius : Extended) : Boolean;

procedure initLaws(var laws : TLaws; size : Longint);
procedure initSimStats(var s : TSimStats; size : Longint);

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

procedure initDelegate(var parliament : TParliament; i : Longint);
begin
  parliament.delegates.delg[i].personalinterestx := getRndminusOnetoOne;
  parliament.delegates.delg[i].collectiveinteresty := getRndminusOnetoOne;

  parliament.delegates.delg[i].party := INDIPENDENT;
end;


// this garuantees that there is no overlapping between parties
function checkParty(var parliament : TParliament; x,y : Extended; party : Longint) : Boolean;
var i : Longint;
begin
  Result := true;
  for i:=1 to party-1 do
       begin
          if distance(x,y,parliament.parties.par[i].centerx,parliament.parties.par[i].centery)<
                      2*parliament.parties.par[i].radius then
                        begin
                          Result := false;
                          Exit;
                        end;
       end;
end;


function initParty(var parliament : TParliament; i : Longint; partyradius : Extended) : Boolean;
var x, y : Extended;
    found : Boolean;
    count : Longint;
begin
  parliament.parties.par[i].radius:= partyradius;

  count := 0;
  repeat
   x := getPartyCenter(partyradius);
   y := getPartyCenter(partyradius);
   found := checkParty(parliament, x,y, i);

   Inc(count);
   if count>MAX_TRIES then
               begin
                 //WriteLn('Unable to allocate parties! Try to reduce their number or their size!');
                 Result := false;
                 Exit;
               end;
  until found;
  parliament.parties.par[i].centerx:= x;
  parliament.parties.par[i].centery:= y;

  parliament.parties.par[i].size := 0;

  Result := true;
end;

procedure addDelegateToParty(var parliament : TParliament; d, party : Longint);
begin
  if parliament.delegates.delg[d].party<>INDIPENDENT then raise Exception.Create('Internal error: delegate was already assigned to a party ('+
                                                                                  IntToStr(d)+','+IntToStr(party)+','+IntToStr(parliament.delegates.delg[d].party)+')');
  parliament.delegates.delg[d].party := party;

  Inc(parliament.parties.par[party].size);
end;

procedure assignDelegatesToParty(var parliament : TParliament; party : Longint);
var x, y : Extended;
    i : Longint;
begin
   x := parliament.parties.par[party].centerx;
   y := parliament.parties.par[party].centery;

   for i:=1 to parliament.delegates.size do
       begin
          if distance(parliament.delegates.delg[i].personalinterestx,parliament.delegates.delg[i].collectiveinteresty,
                      parliament.parties.par[party].centerx,
                      parliament.parties.par[party].centery)<parliament.parties.par[party].radius then
                                    addDelegateToParty(parliament, i, party);
       end;
end;

{
Note: this initialization function garuantees that the delegates are uniformely distributed.
      However, it is not possible to decide how many indipendent delegates there are...
}
function initParliament(var parliament : TParliament; nbdelegates, nbparties : Longint; partyradius : Extended) : Boolean;
var i : Longint;
begin
 if nbdelegates>MAX_DELEGATES then raise Exception.Create('Too many delegetes!');

 parliament.delegates.size:=nbdelegates;
 parliament.parties.size := nbparties;

 for i:=1 to nbdelegates do
     begin
        initDelegate(parliament, i);
     end;

 // we define the parties and their radius so that their surface is mutually exclusive
 for i:=1 to parliament.parties.size do
     begin
        if not initParty(parliament, i, partyradius) then Exit;
     end;

 // we assign delegates to parties, first we use the setting partyradius
 for i:=1 to parliament.parties.size do
     begin
        assignDelegatesToParty(parliament, i);
     end;

 // the remaining unassigned delegates are indipendent
 parliament.indipendents:=0;
 for i:=1 to nbdelegates do
     begin
        if parliament.delegates.delg[i].party=INDIPENDENT then
                    Inc(parliament.indipendents);
     end;

 Result := true;
end;

{
Note: using this function to initialize the parliament allows to specify the nubmer of indipendent delegates. However, the
 delegates are no longer uniformely distributed in the plane (personalinterestx, collectiveinteresty)
}
function initParliamentv2(var parliament : TParliament; nbdelegates, nbindipendents : Longint; partyradius : Extended) : Boolean;
begin

end;

procedure initLaws(var laws : TLaws; size : Longint);
begin
 if size>MAX_LAWS then raise Exception.Create('Too many laws');
 laws.size := size;
end;

procedure initSimStats(var s : TSimStats; size : Longint);
begin
 if size>MAX_LEGISLATURES then raise Exception.Create('Too many legislatures');
 s.size := size;
end;

initialization
  rndgen := TIsaac.Create;

end.

