unit parliamentstructure;

{$mode objfpc}{$H+}

interface

uses
  Classes, SysUtils, isaac;

const MAX_DELEGATES = 300;
      MAX_PARTIES = 5;

type TDelegate = record
    personalinterest,
    collectiveinterest : Extended; // values between -1..1
end;

type TDelegates = record
    delg : Array[1..MAX_DELEGATES] of TDelegate;
    size : Longint;
end;

type TParty  = record
    members : Array[1..MAX_DELEGATES] of Longint; // index pointing to delegates
    size    : Longint;
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
procedure initParliament(nbdelegates, nbindipendents, nbparties : Longint);

implementation

function getRndminusOnetoOne : Extended;
begin
  Result := ( ( rndgen.Val/High(cardinal) )  * 2) - 1;
end;

procedure initDelegate(i : Longint);
begin
  parliament.delegates.delg[i].personalinterest := getRndminusOnetoOne;
  parliament.delegates.delg[i].collectiveinterest := getRndminusOnetoOne;
end;

procedure initParliament(nbdelegates, nbindipendents, nbparties : Longint);
var i : Longint;
begin
 if nbdelegates>MAX_DELEGATES then raise Exception.Create('Too many delegetes!');

 parliament.delegates.size:=nbdelegates;
 for i:=1 to nbdelegates do
     begin
        initDelegate(i);
     end;

end;

initialization
  rndgen := TIsaac.Create;

end.

