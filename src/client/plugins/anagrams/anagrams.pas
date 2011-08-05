unit anagrams;

{$mode objfpc}{$H+}

interface

uses
  Classes, SysUtils, isaac, StrUtils;

const MAXPERMSIZE = 20;
      NONEXISTENT = -1;
      MAXWORDS    = 50000;

      MAXTRIES_ONEWORD  = 500000;
      MAX_SECONDS = 10;

      SECOND = 1/(24*3600);

type TPermutation = Array[1..20] of Byte;

type TGenPermutation = class(TObject)
  public
    constructor Create;
    destructor Destroy;

    procedure init(permsize : Longint);
    function nextPermutation(var a : TPermutation) : Boolean;
    procedure randomPermutation(var a : TPermutation; permsize : Longint);
    function toString(var a : TPermutation; permsize : Longint) : String;
    function applyPerm(var a : TPermutation; permsize : Longint; aword : String) : String;
  private
    permsize_ : Longint;
    a_        : TPermutation;
    nextPerm_ : Boolean;
    isaac_    : TIsaac;
end;


type TWordExistence = class(TObject)
  public
   constructor Create(filename : String);
   destructor destroy();

   function wordExists(aword : String) : Boolean;
  private
    filename_    : String;
    wordsloaded_ : Longint;
    words_       : Array[1..MAXWORDS] of String;
    index_       : Array[1..128] of Array[1..128] of Array[1..128] of Longint;
end;

type TAnagram = class(TObject)
  public
   constructor Create(aword : String);
   constructor Create(aword : String; wexistence : TWordExistence);
   destructor destroy();
   function findAnagram() : String;

  private
   permgen_    : TGenPermutation;
   perm_       : TPermutation;
   length_     : Longint;
   aword_      : String;
   wexistence_ : TWordExistence;
   freeit_     : Boolean;
   maxtries_   : Longint;

   function nextAnagram() : String;
end;

type TTwoWordAnagram = class(TObject)
   function findAnagram(bigword : String) : String;
end;


implementation

constructor TGenPermutation.Create();
begin
  inherited;
  isaac_ := TIsaac.Create;
end;

destructor TGenPermutation.destroy;
begin
  isaac_.Free;
  inherited;
end;

procedure TGenPermutation.init(permsize : Longint);
var i : Longint;
begin
 permsize_ := permsize;
 for i:=1 to permsize do
   a_[i] := i;
 nextPerm_ := true;
end;

procedure TGenPermutation.randomPermutation(var a : TPermutation; permsize : Longint);
var i,n,di : Longint;
    swp    : Byte;
begin
 for i:=1 to permsize do a[i] := i;

{  for i from n downto 2
   do   di ← random element of { 0, ..., i − 1 }
        swap a[di] and a[i − 1]  }
 i:=permsize;
 while (i>0) do
   begin
     di := Trunc(isaac_.Val/MAXINT/2 * (i-1))+1;
     swp := a[di];
     a[di] := a[i];
     a[i] := swp;
     i:=i-2;
   end;

end;

function TGenPermutation.nextPermutation(var a : TPermutation) : Boolean;
var i, k, l, maxk, maxl, q : Longint;
    swp : Byte;
begin
{ algo from wikipedia, generates all permutations // there is also a random version of it
   1. Find the largest index k such that a[k] < a[k + 1]. If no such index exists, the permutation is the last permutation.
   2. Find the largest index l such that a[k] < a[l]. Since k + 1 is such an index, l is well defined and satisfies k < l.
   3. Swap a[k] with a[l].
   4. Reverse the sequence from a[k + 1] up to and including the final element a[n].
}
 Result := nextPerm_;
 if nextPerm_=false then Exit;
 // copying the perm into the array
 for i:=1 to permsize_ do
   a[i] := a_[i];

 // computing next perm
 // step 1.
 maxk := -1;
 for k:=1 to permsize_-1 do
    if a_[k]<a_[k+1] then maxk:=k;
 if (maxk=-1) then
     begin
       nextPerm_ := false;
       Exit;
     end;
 // step 2.
 for l:=maxk+1 to permsize_ do
     if a_[maxk]<a_[l] then maxl:=l;

 // step 3
 swp := a_[maxk];
 a_[maxk] := a_[maxl];
 a_[maxl] := swp;

 // step 4
 q:=0;
 for i:=maxk+1 to ((permsize_-maxk) div 2) + maxk do
    begin
      swp := a_[i];
      a_[i] := a_[permsize_-q];
      a_[permsize_-q] := swp;
      Inc(q);
    end;

end;

function TGenPermutation.toString(var a : TPermutation; permsize : Longint) : String;
var i   : Longint;
    str : String;
begin
  str := '';
  for i:=1 to permsize do
    str := str + IntToStr(a[i]);
  Result := str;
end;

function TGenPermutation.applyPerm(var a : TPermutation; permsize : Longint; aword : String) : String;
var str : String;
    i   : Longint;
begin
 Result := '';
 if length(aword)<>permsize then Exit;
 str := '';
 for i:=1 to permsize do
     str := str + aword[a[i]];

 Result := str;
end;

function faculty(n : Longint) : Longint;
var i : Longint;
begin
  Result := 1;
  for i:=1 to n do
     Result := Result * i;
end;

constructor TAnagram.Create(aword : String; wexistence : TWordExistence);
begin
  inherited Create();
  length_ := length(aword);
  aword_ := aword;
  permgen_ := TGenPermutation.Create();
  permgen_.init(length_);
  wexistence_ := wexistence;
  freeit_ := false;
  maxtries_ := faculty(length_)+1;
  if maxtries_>MAXTRIES_ONEWORD then maxtries_:=MAXTRIES_ONEWORD;
end;


constructor TAnagram.Create(aword : String);
begin
  Create(aword, TWordExistence.Create('wordlist.txt'));
  freeit_ := true;
end;

destructor TAnagram.Destroy();
begin
  permgen_.Free;
  if freeit_ then wexistence_.Free;
  inherited Destroy();
end;

function TAnagram.nextAnagram() : String;
var
    resWord : String;
    i       : Longint;
begin
 resWord := '';
 if permgen_.nextPermutation(perm_) then
   begin
     for i:=1 to length_ do
       resWord := resWord + aword_[perm_[i]];
   end;
 Result := resWord;
end;

function TAnagram.findAnagram() : String;
var tries : Longint;
    aword : String;
begin
  Result := '';
  tries := 0;
  aword := nextAnagram;
  while (aword<>'') and (tries<maxtries_) do
    begin
      if wexistence_.wordExists(aword) then
         begin
           Result := aword;
           Exit;
         end;

      Inc(tries);
      aword := nextAnagram;
    end;
end;

constructor TWordExistence.Create(filename : String);
var count, i, j, k, idx, idx2,idx3 : Longint;
    F                           : TextFile;
    Str                         : String;
begin
  filename_ := filename;
  for k:=1 to 128 do
   for j:=1 to 128 do
    for i:=1 to 128 do index_[i, j, k] := NONEXISTENT;

  AssignFile(F, 'wordlist.txt');
  Reset(F);
  count := 0;
  while (not Eof(F)) and (count<=MAXWORDS) do
   begin
     ReadLn(F, Str);
     Inc(count);
     words_[count] := Str;

     idx  := Ord(Str[1]);
     idx2 := Ord(Str[2]);
     if length(str)>2 then
        idx3 := Ord(Str[3])
     else
        idx3 := 0;
     if (index_[idx, idx2, idx3]=NONEXISTENT) then index_[idx, idx2, idx3]:=count;
   end;
  CloseFile(F);
  wordsloaded_ := count;
end;

destructor TWordExistence.destroy();
begin
end;

function TWordExistence.wordExists(aword : String) : Boolean;
var idx, idx2, idx3, startwith, pos, idxpos : Longint;
    cword : String;
begin
 Result := false;
 aword := lowercase(aword);
 if (length(aword)=1) and ((aword='i') or (aword='a')) then
    Result := true
 else
   begin
     idx  := Ord(aword[1]);
     idx2 := Ord(aword[2]);
     if length(aword)>2 then
        idx3 := Ord(aword[3]) else idx3 := 0;

     startwith := index_[idx, idx2, idx3];
     if startwith <> NONEXISTENT then
           begin
             if idx3=0 then
                  begin
                    Result := true;
                    Exit;
                  end;

             pos := startwith;
             while (pos<wordsloaded_) do
                begin
                  cword := words_[pos];
                  if length(cword)>2 then
                     idxpos := Ord(cword[3])
                  else
                     idxpos := idx3;
                  if idxpos>idx3 then Exit;
                  if cword = aword then
                       begin
                         Result := true;
                         Exit;
                       end;
                  Inc(pos);
                end;
           end;
   end;
end;

function TTwoWordAnagram.findAnagram(bigword : String) : String;
var wexistence   : TWordExistence;
    perm         : TPermutation;
    genperm      : TGenPermutation;
    anagram      : TAnagram;
    l,
    i, split     : Longint;
    word1, word2,
    anaword1, anaword2,
    initword     : String;
    initstamp    : TDateTime;
begin
  Result := '';
  wexistence := TWordExistence.Create('wordlist.txt');
  l := length(bigword);
  genperm := TGenPermutation.Create();

  if (l<6) then Exit;

  initstamp := Now;
  while ((Now-InitStamp)<(MAX_SECONDS*SECOND)) do
     begin
       genperm.randomPermutation(perm,l);
       initword := genperm.applyPerm(perm, l, bigword);
       split := l div 2;
       for i:=split-1 to split+1 do
           begin
             word1 := Copy(bigword,1,split);
             word2 := Copy(bigword,split+1,l);

             anagram := TAnagram.Create(word1, wexistence);
             anaword1 := anagram.findAnagram();
             anagram.Free;
             if anaword1 <> '' then
                  begin
                    anagram := TAnagram.Create(word2, wexistence);
                    anaword2 := anagram.findAnagram();
                    anagram.Free;
                    if anaword2 <> '' then
                         begin
                           Result := anaword1+' '+anaword2;
                           Exit;
                         end;
                  end;
           end;
     end;

  genperm.Free;
  wexistence.Free;
end;

end.

