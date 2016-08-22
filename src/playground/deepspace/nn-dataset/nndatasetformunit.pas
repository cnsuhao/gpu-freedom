unit nndatasetformunit;

{$mode objfpc}{$H+}

interface

uses
  Classes, SysUtils, FileUtil, Forms, Controls, Graphics, Dialogs, StdCtrls,
  ExtCtrls, Spin, csvtables;

type

  { TNNdataSetForm }

  TNNdataSetForm = class(TForm)
    btnLoad: TButton;
    btnSelectInputFileName: TButton;
    btnGenerateDataSet: TButton;
    btnSave: TButton;
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
    edtTrainFilename: TEdit;
    edtSeparator: TEdit;
    edtTestFilename: TEdit;
    InputBox: TGroupBox;
    lblInputFilename2: TLabel;
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

  private
    inputCSV_ : TCSVTable;

    function getSeparator(): AnsiString;
  public
    { public declarations }
  end;

var
  NNdataSetForm: TNNdataSetForm;

implementation

{$R *.lfm}

{ TNNdataSetForm }

function TNNdataSetForm.getSeparator(): AnsiString;
begin
  if rbSeparatorEdit.Checked then
      Result := edtSeparator.Text
  else
  if rbSeparatorTab.Checked then
      Result := Chr(9)
  else
    ShowMessage('Internal error: undefined radio box');
end;

procedure TNNdataSetForm.btnSelectInputFileNameClick(Sender: TObject);
var i : Longint;
begin
   if myOpenDialog.Execute then
        begin
          edtInputFileName.Text := myOpenDialog.FileName;
          edtTrainFileName.Text := extractFileName(myOpenDialog.FileName)+'.train';
          edtTestFileName.Text := extractFileName(myOpenDialog.FileName)+'.test';

          inputCSV_ :=  TCSVTable.Create(myOpenDialog.FileName, getSeparator());
          inputCSV_.loadInMemory;

          for i:=1 to inputCSV_.totalfields_ do
              begin
                   cbFieldInput1.Items.Add(inputCSV_.fields_[i]);
                   cbFieldInput2.Items.Add(inputCSV_.fields_[i]);
                   cbFieldInput3.Items.Add(inputCSV_.fields_[i]);
                   cbFieldInput4.Items.Add(inputCSV_.fields_[i]);
                   cbFieldInput5.Items.Add(inputCSV_.fields_[i]);
                   cbFieldInput6.Items.Add(inputCSV_.fields_[i]);
                   cbFieldInput7.Items.Add(inputCSV_.fields_[i]);
                   cbFieldInput8.Items.Add(inputCSV_.fields_[i]);
                   cbFieldInput9.Items.Add(inputCSV_.fields_[i]);
                   cbFieldInput10.Items.Add(inputCSV_.fields_[i]);
                   cbFieldInput11.Items.Add(inputCSV_.fields_[i]);
                   cbFieldInput12.Items.Add(inputCSV_.fields_[i]);
                   cbFieldInput13.Items.Add(inputCSV_.fields_[i]);
                   cbFieldInput14.Items.Add(inputCSV_.fields_[i]);

                   cbFieldOutput1.Items.Add(inputCSV_.fields_[i]);
                   cbFieldOutput2.Items.Add(inputCSV_.fields_[i]);
                   cbFieldOutput3.Items.Add(inputCSV_.fields_[i]);
                   cbFieldOutput4.Items.Add(inputCSV_.fields_[i]);
                   cbFieldOutput5.Items.Add(inputCSV_.fields_[i]);
                   cbFieldOutput6.Items.Add(inputCSV_.fields_[i]);
                   cbFieldOutput7.Items.Add(inputCSV_.fields_[i]);
                   cbFieldOutput8.Items.Add(inputCSV_.fields_[i]);
              end;


        end;
end;


end.

