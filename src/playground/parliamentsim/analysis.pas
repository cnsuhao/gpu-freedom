unit analysis;

{ Analytic function taken
  from Le Scienze 1/2013 (Italian version of Scientific American)
  in article
  "L'efficienza del caso"

  Gli autori hanno applicato lo studio dei sistemi complessi all'efficienza di un ipotetico Parlamento, definita come
  il prodotto della percentuale di proposte di legge approvate moltiplicata per il benessere sociale assicurato da quelle leggi.
  Le simulazioni ad agenti su cui si è basato lo studio hanno mostrato che l'efficienza di questo Parlamento virtuale raggiunge
  il massimo con un numero ottimale di parlamentari estratti a sorte e non aderenti ad alcun partito.
  Il risultato ha mostrato anche che i processi basati sul caso, fondamentali in tanti problemi fisici, sono utili anche in
  campo socioeconomico tramite strategie che prevedono scelte casuali

  by Alessandro Pluchino, Andrea Rapisarda, Cesare Garofalo, Salvatore Spagano e Maurizio Caserta
}

{$mode objfpc}{$H+}

interface

uses
  Classes, SysUtils;

function optimalIndipendentsNumber(delegates : Longint; majorPartyPercentage : Extended) : Extended;

implementation

// this formula is taken from the article above
function optimalIndipendentsNumber(delegates : Longint; majorPartyPercentage : Extended) : Extended;
var nominator, denominator : Extended;
begin
  nominator   := 2*delegates - (4*delegates*majorPartyPercentage)+ 4;
  denominator := 1- (4*majorPartyPercentage);

  Result := nominator/denominator;
end;

end.

