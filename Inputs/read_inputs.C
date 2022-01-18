#include <fstream>

TFile *file_inputs = nullptr;

void TH1D_to_txt(const char* filename_, const char* th_name_);

int read_inputs ()
{
    file_inputs = new TFile("JUNOInputs2021_05_28_noTF2.root","read");

//    TH1D_to_txt("JUNOInputs2021_05_28_noTF2.root","IBDXsec_StrumiaVissani");
//    TH1D_to_txt("JUNOInputs2021_05_28_noTF2.root","IBDXsec_VogelBeacom_DYB");
    // corrections
//    TH1D_to_txt("JUNOInputs2021_05_28_noTF2.root","SNF_FluxRatio");
//    TH1D_to_txt("JUNOInputs2021_05_28_noTF2.root","SNF_VisibleSpectrum");
//    TH1D_to_txt("JUNOInputs2021_05_28_noTF2.root","NonEq_FluxRatio");
//    TH1D_to_txt("JUNOInputs2021_05_28_noTF2.root","NonEq_VisibleSpectrum");
//    TH1D_to_txt("JUNOInputs2021_05_28_noTF2.root","DYBFluxBump_ratio");
    // non linearity
    TH1D_to_txt("JUNOInputs2021_05_28_noTF2.root","positronScintNL");
    TH1D_to_txt("JUNOInputs2021_05_28_noTF2.root","positronScintNLpull0");
    TH1D_to_txt("JUNOInputs2021_05_28_noTF2.root","positronScintNLpull1");
    TH1D_to_txt("JUNOInputs2021_05_28_noTF2.root","positronScintNLpull2");
    TH1D_to_txt("JUNOInputs2021_05_28_noTF2.root","positronScintNLpull3");
    // backgrounds
//    TH1D_to_txt("JUNOInputs2021_05_28_noTF2.root","AccBkgHistogramAD");
//    TH1D_to_txt("JUNOInputs2021_05_28_noTF2.root","FnBkgHistogramAD");
//    TH1D_to_txt("JUNOInputs2021_05_28_noTF2.root","Li9BkgHistogramAD");
//    TH1D_to_txt("JUNOInputs2021_05_28_noTF2.root","AlphaNBkgHistogramAD");
//    TH1D_to_txt("JUNOInputs2021_05_28_noTF2.root","GeoNuHistogramAD");

    file_inputs->Close();

    return 0;
}

void TH1D_to_txt(const char* filename_, const char* th_name_)
{

    printf("Creating %s.csv \n", th_name_);

    TH1D *h_appo = (TH1D*) file_inputs->Get(th_name_);
    int n_bins = h_appo->GetNbinsX();

    std::fstream out;
    out.open(Form("%s.csv", th_name_), std::fstream::out);

    for (int j=0; j < n_bins; j++)
        out << h_appo->GetBinCenter(j) << "," << h_appo->GetBinContent(j) << endl;

    out.close();
}
