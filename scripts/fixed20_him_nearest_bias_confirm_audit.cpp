#include "data_loader.hpp"
#include "experiment_engine.hpp"
#include "extended_features.hpp"

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <vector>

static void adjust(std::vector<double>& p, double neither_bias, double him_bias, int nearest_class) {
    double z[3], mx = -1e300, sum = 0.0;
    for (int c=0;c<3;++c) { z[c]=std::log(std::max(1e-12,p[c]))+(c==2?neither_bias:0.0)+(nearest_class>=0&&c==nearest_class?him_bias:0.0); mx=std::max(mx,z[c]); }
    for (int c=0;c<3;++c) { p[c]=std::exp(z[c]-mx); sum+=p[c]; }
    for (double& x:p) x/=sum;
}

static void score(msar::RepeatedCVResult& result) {
    const std::vector<double> weights={1.0,1.0,1.0};
    result.acc_mean=result.macro_f1_mean=result.log_loss_mean=0.0;
    for (auto& fold:result.folds) {
        std::vector<int> pred(fold.fold_labels.size(),0);
        for(size_t i=0;i<pred.size();++i){ if(fold.fold_probabilities[i][1]>fold.fold_probabilities[i][pred[i]])pred[i]=1; if(fold.fold_probabilities[i][2]>fold.fold_probabilities[i][pred[i]])pred[i]=2; }
        fold.eval_res=msar::evaluatePredictions(fold.fold_labels,pred,fold.fold_probabilities,weights);
        fold.accuracy=fold.eval_res.accuracy; fold.macro_f1=fold.eval_res.macro_f1; fold.log_loss=fold.eval_res.weighted_loss;
        result.acc_mean+=fold.accuracy; result.macro_f1_mean+=fold.macro_f1; result.log_loss_mean+=fold.log_loss;
    }
    const double n=static_cast<double>(result.folds.size()); result.acc_mean/=n; result.macro_f1_mean/=n; result.log_loss_mean/=n;
}

static msar::RepeatedCVResult blendAndAdjust(const msar::RepeatedCVResult& constant,
    const msar::RepeatedCVResult& cosine,const std::vector<msar::GapRecord>& data,
    const std::vector<unsigned int>& seeds,double him_bias) {
    msar::RepeatedCVResult out=constant; size_t fi=0;
    for(unsigned int seed:seeds){ const auto folds=msar::createStratifiedFolds(data,5,seed);
        for(size_t f=0;f<folds.size();++f,++fi){ auto& fold=out.folds[fi];
            for(size_t i=0;i<fold.fold_probabilities.size();++i){
                for(int c=0;c<3;++c) fold.fold_probabilities[i][c]=0.5*(constant.folds[fi].fold_probabilities[i][c]+cosine.folds[fi].fold_probabilities[i][c]);
                adjust(fold.fold_probabilities[i],-0.10,0.0,-1);
                const auto& rec=data[folds[f].val_indices[i]];
                const long long da=std::llabs(rec.pronoun_offset-rec.a_offset);
                const long long db=std::llabs(rec.pronoun_offset-rec.b_offset);
                const int nearest=msar::toLowerString(rec.pronoun)=="him"?(da<db?0:(db<da?1:-1)):-1;
                adjust(fold.fold_probabilities[i],0.0,him_bias,nearest);
            }
        }
    }
    score(out); return out;
}

int main(){
    const auto dev=msar::loadGapDataset("data/raw/gap-development.tsv"); const auto val=msar::loadGapDataset("data/raw/gap-validation.tsv");
    std::vector<msar::GapRecord> data=dev; data.insert(data.end(),val.begin(),val.end());
    msar::ExtendedConfig features; features.use_url=true; features.use_syntax=true; features.use_v2x_matcher=true;
    msar::ExperimentConfig cfg; cfg.name="fixed20_him_nearest_bias_confirm_audit"; cfg.feature_dim=features.getTotalDimension(); cfg.hidden_dim=32; cfg.output_dim=3;
    cfg.learning_rate=0.0012; cfg.class_weight_alpha=0.24; cfg.l2_lambda=0.0001; cfg.batch_size=32; cfg.max_epochs=20;
    cfg.use_swap_aug=true; cfg.use_tta=true; cfg.use_swa=true; cfg.swa_epochs=5; cfg.ensemble_members=10;
    const std::vector<unsigned int> seeds={10300,10301,10302};
    const auto constant=msar::runRepeatedCVExtended(data,features,cfg,seeds); cfg.use_cosine_lr=true;
    const auto cosine=msar::runRepeatedCVExtended(data,features,cfg,seeds);
    auto base=blendAndAdjust(constant,cosine,data,seeds,0.0);
    auto candidate=blendAndAdjust(constant,cosine,data,seeds,-0.10);
    const auto paired=msar::comparePaired(base,candidate);
    std::cout<<std::fixed<<std::setprecision(6)<<"N="<<data.size()<<" seeds=10300,10301,10302 folds=15 test_loaded=no\n";
    std::cout<<"base,"<<base.acc_mean<<','<<base.macro_f1_mean<<','<<base.log_loss_mean<<'\n';
    std::cout<<"him_nearest_bias_minus_0.10,"<<candidate.acc_mean<<','<<candidate.macro_f1_mean<<','<<candidate.log_loss_mean<<'\n';
    std::cout<<"paired_loss_gain,se,delta_accuracy,delta_macro_f1,accepted,"<<paired.mean_loss_diff<<','<<paired.se_loss_diff<<','<<paired.delta_accuracy<<','<<(candidate.macro_f1_mean-base.macro_f1_mean)<<','<<(paired.accepted?"yes":"no")<<'\n';
    return 0;
}
