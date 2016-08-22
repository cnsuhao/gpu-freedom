unit nndatasetformunit;

{$mode objfpc}{$H+}

interface

uses
  Classes, SysUtils, FileUtil, Forms, Controls, Graphics, Dialogs, StdCtrls,
  ExtCtrls, Spin, csvtables;

type

  { TNNdataSetForm }

  TNNdataSetForm = class(TForm)
    btnSelectInputFileName: TButton;
    btnSelectOutputFileName: TButton;
    btnGenerateDataSet: TButton;
    cbFieldInput1: TComboBox;
    cbFieldInput10: TComboBox;
    cbFieldInput11: TComboBox;
    cbFieldInput12: TComboBox;
    cbFieldInput13: TComboBox;
    cbFieldInput14: TComboBox;
    cbFieldOutput1: TComboBox;
    cbFieldInput2: TComboBox;
    cbFieldInput3: TComboBox;
    cbFieldInput4: TComboBox;
    cbFieldInput5: TComboBox;
    cbFieldInput6: TComboBox;
    cbFieldInput7: TComboBox;
    cbFieldInput8: TComboBox;
    cbFieldInput9: TComboBox;
    cbFieldOutput2: TComboBox;
    cbFieldOutput3: TComboBox;
    cbFieldOutput4: TComboBox;
    cbFieldOutput5: TComboBox;
    cbFieldOutput6: TComboBox;
    cbFieldOutput7: TComboBox;
    cbFieldOutput8: TComboBox;
    cbRandomizeRows: TCheckBox;
    edtInputFilename: TEdit;
    edtOutputFilename: TEdit;
    edtSeparator: TEdit;
    InputBox: TGroupBox;
    lblPercentOfTestSet: TLabel;
    lblEndOffset1: TLabel;
    lblFieldname1: TLabel;
    lblStartOffset: TLabel;
    lblFieldname: TLabel;
    lblSeparator: TLabel;
    lblEndOffset: TLabel;
    lblStartOffset1: TLabel;
    OutputBox: TGroupBox;
    lblInputFilename: TLabel;
    lblInputFilename1: TLabel;
    myOpenDialog: TOpenDialog;
    rbSeparatorEdit: TRadioButton;
    rbSeparatorTab: TRadioButton;
    spnOffsetEnd10: TSpinEdit;
    spnOffsetEnd11: TSpinEdit;
    spnOffsetEnd12: TSpinEdit;
    spnOffsetEnd13: TSpinEdit;
    spnOffsetEnd14: TSpinEdit;
    spnOffsetStart15: TSpinEdit;
    spnOutOffsetEnd1: TSpinEdit;
    spnOffsetEnd2: TSpinEdit;
    spnOffsetEnd3: TSpinEdit;
    spnOffsetEnd4: TSpinEdit;
    spnOffsetEnd5: TSpinEdit;
    spnOffsetEnd6: TSpinEdit;
    spnOffsetEnd7: TSpinEdit;
    spnOffsetEnd8: TSpinEdit;
    spnOffsetEnd9: TSpinEdit;
    spnOffsetStart1: TSpinEdit;
    spnOffsetEnd1: TSpinEdit;
    spnOffsetStart10: TSpinEdit;
    spnOffsetStart11: TSpinEdit;
    spnOffsetStart12: TSpinEdit;
    spnOffsetStart13: TSpinEdit;
    spnOffsetStart14: TSpinEdit;
    spnOutOffsetEnd2: TSpinEdit;
    spnOutOffsetEnd3: TSpinEdit;
    spnOutOffsetEnd4: TSpinEdit;
    spnOutOffsetEnd5: TSpinEdit;
    spnOutOffsetEnd6: TSpinEdit;
    spnOutOffsetEnd7: TSpinEdit;
    spnOutOffsetEnd8: TSpinEdit;
    spnOutOffsetStart1: TSpinEdit;
    spnOffsetStart2: TSpinEdit;
    spnOffsetStart3: TSpinEdit;
    spnOffsetStart4: TSpinEdit;
    spnOffsetStart5: TSpinEdit;
    spnOffsetStart6: TSpinEdit;
    spnOffsetStart7: TSpinEdit;
    spnOffsetStart8: TSpinEdit;
    spnOffsetStart9: TSpinEdit;
    spnOutOffsetStart2: TSpinEdit;
    spnOutOffsetStart3: TSpinEdit;
    spnOutOffsetStart4: TSpinEdit;
    spnOutOffsetStart5: TSpinEdit;
    spnOutOffsetStart6: TSpinEdit;
    spnOutOffsetStart7: TSpinEdit;
    spnOutOffsetStart8: TSpinEdit;
    procedure btnSelectInputFileNameClick(Sender: TObject);
    procedure btnSelectOutputFileNameClick(Sender: TObject);
  private
    { private declarations }
  public
    { public declarations }
  end;

var
  NNdataSetForm: TNNdataSetForm;

implementation

{$R *.lfm}

{ TNNdataSetForm }

procedure TNNdataSetForm.btnSelectInputFileNameClick(Sender: TObject);
begin
   if myOpenDialog.Execute then
          edtInputFileName.Text := myOpenDialog.FileName;
end;

procedure TNNdataSetForm.btnSelectOutputFileNameClick(Sender: TObject);
begin
   if myOpenDialog.Execute then
          edtOutputFileName.Text := myOpenDialog.FileName;
end;

end.

